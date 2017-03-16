#! .env/bin/python

import sys
from a2audio.training_lib import run_training


USAGE = """Runs a training job.
{prog} job_id
    job_id - job id in database
""".format(prog=sys.argv[0])

def main(argv):
    if len(argv) < 2:
        print USAGE
        sys.exit()
    else:
        jobId = int(str(argv[1]).strip("'"))
        retVal = run_training(jobId)
        if retVal:
            print 'end'
        else:
            print 'err'

if __name__ == '__main__':
    main(sys.argv)

