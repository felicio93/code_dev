#!/usr/bin/env python3

'''
Automatically submit and monitor the run status.
Restart the run from the latest hotstart point if the run stops.

(1) Copy this script, run_test and run_comb into rundir

(2) Inside rundir: prep inputs as before
    The "hotstart.nc" (if it exists under ihot=1 or 2) will be overwritten by this script by symlinks ("ln -sf"); 
    if you want to keep it, "mv hotstart.nc hotstart.nc.0" then "ln -s hotstart.nc.0 hotstart.nc"

(3) Under the run dir,
      "python auto_hotstart.py >& scrn.out"
    or
      "./auto_hotstart.py > scrn.out &"
      "tail -f scrn.out"
    The latter prints jobs status and run speeds to the screen continuously.

    This can be done at any stage of the run, for example:
    * Before a run is launched, i.e., the script will submit the initial job and monitor its progress;
    * After the run is launched manually;
    * After the run is interrupted, e.g., due to time limit or manual scancel

    Note: the script will use the last part of the current dir as runid, e.g., the runid of "RUN13a" or "R13a" will be "13a"
          , make sure you don't have duplicate run ids.

(4) Sometimes a run can hang, e.g., on Hercules, the script will automatically detect this and interrupt the run then relaunch from
    the nearest hotstarting point

(5) Start by specifying inputs below, between the lines "inputs" and "end inputs"
'''

import os
import time
import subprocess
import glob
import re
from datetime import datetime


# ----------------inputs---------------------
JOB_SCHEDULER = 'slurm'  # 'pbs' or 'slurm'

rundir = os.getcwd()  # use os.getcwd if launch from rundir; otherwise specify a string yourself
last_stack = None  # if None, the script will try to find the last stack number in the file "param.nml"
                   # make sure the run can finish the specified rnday in param.nml (i.e., the forcing covers the whole period);
                   # otherwise, change the "rnday" in param.nml or specify another number here
# ----------------end inputs---------------------


# ---------------------  embedded functions -----------------------
def create_empty_files(file_names):
  """Creates empty files and returns a list of their names.

  Args:
      file_names: A list of file names to create.

  Returns:
      A list of the created file names.
  """
  created_files = []
  for file_name in file_names:
    try:
      with open(file_name, 'x') as f:  # Use 'x' mode for exclusive creation
        pass  # 'pass' does nothing, creating an empty file
      created_files.append(file_name)
    except FileExistsError:
      print(f"File '{file_name}' already exists.")
  return created_files

def decor_print(msg, prefix='', suffix=''):
    '''
    Decorative print with prefix and suffix
    '''
    print(f'{prefix}{msg}{suffix}', flush=True)


def Replace_string_in_file(fname, str_orig_pattern, str_replace):
    '''
    Match and replace a string in a file
    '''
    if '~' in fname:
        fname = fname.replace('~', os.path.expanduser('~'))
    fname = os.path.abspath(fname)
    with open(fname, "rt", encoding='utf-8') as fin:
        with open("tmp.txt", "wt", encoding='utf-8') as fout:
            for line in fin:
                fout.write(re.sub(str_orig_pattern, str_replace, line))
    os.system(f"mv tmp.txt {fname}")


def ReplaceJobName(fname, job_name, job_scheduler='slurm'):
    import fileinput

    if job_scheduler == 'slurm':
        pattern = r"(SBATCH\s+-J\s+)(\S+)"
    else:
        raise ValueError('job_scheduler must be either "slurm"; pbs not implemented yet')

    replacement = rf'SBATCH -J {job_name}'

    # Use fileinput to edit the file in place
    match_found = False
    with fileinput.FileInput(fname, inplace=True) as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                match_found = True
            modified_line = re.sub(pattern, replacement, line)
            print(modified_line, end='')

    if not match_found:
        raise ValueError(f'Job name specification not found in {fname}')


def Get_hotstart_step(run_out_dir):
    if '~' in run_out_dir:
        run_out_dir = run_out_dir.replace('~', os.path.expanduser('~'))
    run_out_dir = os.path.abspath(run_out_dir)

    hot_files = glob.glob(f"{run_out_dir}/hotstart_000000_*.nc")

    hot_steps = []
    for hot_file in hot_files:
        sub_str = hot_file.split("_")
        sub_str = sub_str[-1].split(".")
        hot_steps.append(int(sub_str[0]))

    hot_steps.sort()
    return hot_steps


def Get_var_from_file(fname, var_name, reverse_search=False):
    if '~' in fname:
        fname = fname.replace('~', os.path.expanduser('~'))
    fname = os.path.abspath(fname)

    pattern = fr'{var_name}\s*=\s*(\d+)'
    with open(fname, "rt") as fin:
        lines = fin.readlines()  # Read all lines into a list

    if reverse_search:
        lines = [line for line in reversed(lines)]

    for line in lines:
        match = re.search(pattern, line)
        if match:
            var_value = match.group(1)  # Extract the matched group
            return var_value

    print(f'Variable {var_name} not found in {fname}')
    return None

def Get_var_from_wwmfile(fname, var_name, reverse_search=False):
    if '~' in fname:
        fname = fname.replace('~', os.path.expanduser('~'))
    fname = os.path.abspath(fname)

    #pattern = fr'{var_name}\s*=\s*(\d+)'
    pattern = re.compile(rf'\s*{var_name}\s*=\s*(\S.*\S|\S)')
    with open(fname, "rt") as fin:
        lines = fin.readlines()  # Read all lines into a list

    if reverse_search:
        lines = [line for line in reversed(lines)]

    for line in lines:
        match = re.search(pattern, line)
        if match:
            var_value = match.group(1)  # Extract the matched group
            return var_value

    print(f'Variable {var_name} not found in {fname}')
    return None
# --------------------- end embedded functions -----------------------


def auto_hotstart(job_scheduler='slurm', rundir=None, last_stack=None):
    '''
    Main function to submit and monitor the run status
    '''
    my_print_prefix = ''
    my_print_suffix = ''

    # ---------------------  get background information -----------------------
    # options not exposed to users
    run_id_normal = ''
    rundir_normal = ''  # f'/scratch1/02786/{user_name}/RUN{run_id_normal}/'
    # end options not exposed to users

    user_name = os.getlogin()  # getlogin() or specify a string yourself

    if job_scheduler == 'slurm':
        batch_cmd = 'sbatch'
        queue_query_str = f"squeue -u {user_name}"
    elif job_scheduler == 'pbs':
        batch_cmd = 'qsub'
        queue_query_str = f"qstat -u {user_name}"
    else:
        raise ValueError('job_scheduler must be either "slurm" or "pbs"')

    run_id = os.path.basename(rundir)
    if len(run_id) > 8:
        raise ValueError('run_id must be 8 characters or less, otherwise it may be truncated by the job scheduler')
    decor_print(f'RUN job name : {run_id}')

    combine_job_name = f'{run_id}_cmb'
    if "RUN" in combine_job_name:
        combine_job_name = combine_job_name.replace("RUN", "")  # shorten the name if it contains "RUN"
    if len(combine_job_name) > 8:
        raise ValueError('combine_job_name must be 8 characters or less, otherwise it may be truncated by the job scheduler')
    decor_print(f'Combine job name : {combine_job_name}')

    # get the last stack number in the file "param.nml"
    # Define the regular expression pattern to match "rnday =" followed by a number
    if last_stack is None:
        rnday = float(Get_var_from_file(f'{rundir}/param.nml', 'rnday'))
        dt = float(Get_var_from_file(f'{rundir}/param.nml', 'dt'))
        ihfskip = int(Get_var_from_file(f'{rundir}/param.nml', 'ihfskip'))

        last_stack = int(rnday * 86400 / dt / ihfskip)

    # ----------------- prepare job names in batch script --------------------
    ReplaceJobName(f'{rundir}/run_test', run_id, job_scheduler)
    ReplaceJobName(f'{rundir}/run_comb', combine_job_name, job_scheduler)

    os.chdir(f'{rundir}')

    # --------------------- initialize empty files -----------------------
    file_names_to_create = [f'{rundir}/outputs/staout_{i}' for i in range(1,10)]+[f'{rundir}/outputs/flux.out']
    create_empty_files(file_names_to_create)
    # create backups:
    os.system(f'cp {rundir}/wwminput.nml {rundir}/wwminput0.nml')
    os.system(f'cp {rundir}/param.nml {rundir}/param0.nml')

    # --------------------- monitor the run -----------------------    
    previous_time_step = -1  # initialize
    num_run = 1 #counts how many times the model was hotstarted 
    while (not os.path.exists(f'{rundir}/outputs/schout_000000_{last_stack+1}.nc')) and (not os.path.exists(f'{rundir}/outputs/out2d_{last_stack}.nc')):
        print(f'\n{"%" * 80}\n  Local time: {datetime.now()}\n{"%" * 80}\n')  # step header

        job_status = subprocess.getoutput(queue_query_str)
        print(job_status)

        # extract the line corresponding to the current run_id
        current_job_status = None
        for line in job_status.splitlines():
            if run_id in line:
                current_job_status = line.strip()
                print(f'\n{"-"*100}\ncurrent job status: {current_job_status}\n{"-"*100}\n') 
                break

        if rundir_normal != '':
            if re.search(rf"{run_id_normal}\s+{user_name}\s+R", current_job_status) is not None:
                decor_print(f'{run_id_normal} running')
            elif re.search(rf"{run_id_normal}\s+{user_name}\s+PD", current_job_status) is not None:
                decor_print(f'{run_id_normal} pending')
            else:
                decor_print(f'{run_id_normal} does not exist')

        if current_job_status is not None:
            job_id = re.search(r'\b\d+\b', current_job_status).group()
            decor_print(f'job_id: {job_id}')

            # check run status
            if re.search(rf"{run_id}\s+{user_name}\s+R", current_job_status) is not None:  # job id in squeue

                # make sure mirror.out is written after the job starts;
                # if it is not written after 60s, then the run probably hangs
                time.sleep(60)

                # see if the run hangs
                hanged = False
                # check if mirror.out is written
                if not os.path.exists(f'{rundir}/outputs/mirror.out'):
                    hanged = True
                else:
                    time_step = Get_var_from_file(f'{rundir}/outputs/mirror.out', 'TIME STEP', reverse_search=True)
                    if time_step is not None:
                        time_step = int(time_step)
                        if time_step == previous_time_step:
                            decor_print(f'{run_id} hangs at Time Step, killing job and relaunch without changing hotstart')
                            hanged = True
                        else:
                            decor_print(f'Run advancing, time step: {time_step}')
                            previous_time_step = time_step
                if hanged:
                    print(f'scancel {job_id}')
                    os.system(f'scancel {job_id}')
                    time.sleep(20)  # wait for scancel
                    print(f'{batch_cmd} run_test')
                    os.system(f'{batch_cmd} run_test')
                    decor_print(f'{run_id} hanged and was resubmitted ...')
                else:
                    decor_print(f'{run_id} running, wait ...')
            else:
                previous_time_step = -1  # resetting for unforeseen cases
                decor_print(f'{run_id} queueing, wait ...')
            time.sleep(120)
        else:  # i.e., current_job_status is None (job id not in squeue)
            # check if the run finishes normally
            # open a file and check if the last line contains "Run completed successfully"
            if os.path.exists(f'{rundir}/outputs/mirror.out'):
                with open(f'{rundir}/outputs/mirror.out', 'r') as file:
                    lines = file.readlines()  # Read all lines in the file
                    last_line = lines[-1] if lines else ''  # Get the last line if the file is not empty
                    # Check if the last line contains the desired phrase
                    if "Run completed successfully" in last_line:
                        decor_print("The last line indicates that the run completed successfully.")
                        break
                    else:
                        decor_print("The last line does not indicate a successful completion, try combining the last hotstart.nc the restart the run.")
                # combine hotstart
                hot_steps = Get_hotstart_step(f'{rundir}/outputs/')
                if len(hot_steps) == 0:
                    raise Exception('No hotstart files generated before run stopped.')
                hot_step = hot_steps[-1]

                hot_combined = f'{rundir}/outputs/hotstart_it={hot_step}.nc'
                decor_print(f'{run_id} stopped, last hotstart to combine: {hot_combined}')

                os.chdir(f'{rundir}/outputs/')
                Replace_string_in_file(f'{rundir}/run_comb', '-i 0000', f'-i {hot_step}')

                decor_print(f'{batch_cmd} run_comb')
                os.system(f'{batch_cmd} {rundir}/run_comb')

                # restore run_cmb template
                os.system(f'cat {rundir}/run_comb')
                Replace_string_in_file(f'{rundir}/run_comb', f'-i {hot_step}', '-i 0000')

                # wait for combine hotstart to finish
                while combine_job_name in subprocess.getoutput(queue_query_str):
                    time.sleep(20)
                    decor_print(f'waiting for job {combine_job_name} to finish')
                if not os.path.exists(hot_combined):
                    raise Exception(f'Failed generating {hot_combined}')

                time.sleep(20)

                # link combined hotstart.nc
                os.chdir(f'{rundir}')
                decor_print(f'linking {hot_combined}')
                os.system(f'rm {rundir}/hotstart.nc')
                os.symlink(hot_combined, f'{rundir}/hotstart.nc')
                
                # dealing with the paired normal run
                if rundir_normal != '':
                    if re.search(rf"{run_id_normal}\s+{user_name}\s+R", current_job_status) is not None:
                        decor_print(f'{run_id_normal} has started, skipping syncing files with it')
                    else:
                        decor_print(f'copying staout* and *.out to {rundir_normal}')
                        os.system(f'cp {rundir}/outputs/staout* {rundir_normal}/outputs/')
                        os.system(f'cp {rundir}/outputs/*.out {rundir_normal}/outputs/')

                Replace_string_in_file(f'{rundir}/param.nml', r'ihot\s*=\s*\d+', 'ihot = 2')
                time.sleep(20)

                ## Prepare WWM hotstart:
                decor_print(f'Preparing WMM hotstart for the {num_run} time')

                os.system(f'mkdir -p {rundir}/wwm_files')
                wwm_hotout = Get_var_from_wwmfile(f'{rundir}/wwminput.nml', 'FILEHOT_OUT')
                os.system(f'mv {rundir}/{wwm_hotout}* {rundir}/wwm_files/')
                wwm_hotout_files = glob.glob(f"{rundir}/wwm_files/*")
                for wwm_hotout_file in wwm_hotout_files:
                    wwm_hotin_file = os.path.basename(wwm_hotout_file).replace('out', 'in')
                    wwm_hotin_file = os.path.join(rundir, wwm_hotin_file)
                    os.symlink(wwm_hotout_file, wwm_hotin_file)

                # Get the list of ocean_time in the hotstart
                variable_name = "ocean_time_str" 
                command = f"ncdump -v {variable_name} {wwm_hotout_files[0]}"
                with os.popen(command) as stream:
                    hot_time = stream.read()
                hot_time = hot_time.split(f'{variable_name} =')[-1].split(';')[0]
                hot_time = re.sub(r"\s+", "", hot_time)
                hot_time = hot_time.replace('"','')
                hot_time_s = [i for i in hot_time.split(',') if i != ""]
                hot_time_f = [float(i) for i in hot_time.split(',') if i != ""]
                idx_max_hot = hot_time_f.index(max(hot_time_f))

                decor_print(f'The following ocean_time_str were found in the wwm hotstart: {hot_time_s}, selecting the largest: {hot_time_s[idx_max_hot]}')

                # Updated wwmintput.nml: '\s*{var_name}\s*=\s*(\S.*\S|\S)'
                Replace_string_in_file(f'{rundir}/wwminput.nml', r'\s*LHOTR\s*=\s*(\S.*\S|\S)', '    LHOTR = .true.!')
                Replace_string_in_file(f'{rundir}/wwminput.nml', r'\s*BEGTC\s*=\s*(\S.*\S|\S)', f'    BEGTC = {hot_time_s[idx_max_hot]}!')
                Replace_string_in_file(f'{rundir}/wwminput.nml', r'\s*IHOTPOS_IN\s*=\s*(\S.*\S|\S)', f'    IHOTPOS_IN = {int(idx_max_hot+1)}!')  
                            

                ## WWM stations, wwm_sta
                os.system(f'mkdir -p {rundir}/wwm_stations')
                for file_path in glob.glob(os.path.join(rundir, 'wwm_sta_*')):
                    file_name = os.path.basename(file_path)
                    new_file_name = f"{num_run}_{file_name}" 
                    # Construct the new file path in the destination directory
                    destination_path = os.path.join(f'{rundir}/wwm_stations', new_file_name)  
                    os.rename(file_path, destination_path)

                
                num_run += 1

            # -- end if  mirror.out exists

            # submit new job
            os.chdir(f'{rundir}')
            decor_print(f'{batch_cmd} run_test')
            os.system(f'{batch_cmd} run_test')

        # restore template run_test
            # Replace_string_in_file('~/bin/run_test', f'RUN{run_id}', 'RUNxxx')

    os.system(f'rm {rundir}/hotfile*')
    os.system(f'rm {rundir}/outputs/hotstart_0*')
    decor_print('Task completed. ')


if __name__ == '__main__':
    auto_hotstart(job_scheduler=JOB_SCHEDULER, rundir=rundir, last_stack=last_stack)
