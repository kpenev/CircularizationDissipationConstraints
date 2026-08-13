import sys
from POET.solver import poet_solver
from multiprocessing_util import setup_process

def fitall(systemname,systempath):
    params = {
        "type": 'blank',
        "epochs": 350,
        "batch_size": 50,
        "verbose": 2,
        "retrain": False,
        "threshold": 2000,
        "path_to_store": systempath,
        "version": systemname,
        "features": [True, True, True, True, True, True, True, True, True, True]
    }

    def dofit(param_type,params):
        params['type'] = param_type
        model = poet_solver.POET_IC_Solver(**params)
        length = model.data_length()
        if length > params['threshold']:
            model.just_fit()

    dofit('1d_period',params)
    params['features'].append(True)
    dofit('2d_period',params)
    dofit('2d_eccentricity',params)

if __name__ == "__main__":
    systemname = str(sys.argv[1])
    systempath = str(sys.argv[2])

    setup_process(
                    fname_datetime_format='%Y%m%d%H%M%S',
                    system=systemname,
                    std_out_err_fname='/scratch/08402/vortebo/ls6/output/W19/nn_data/training_output/{task}/{system}_{now}_{pid:d}.outerr',
                    logging_fname='/scratch/08402/vortebo/ls6/output/W19/nn_data/training_output/{task}/{system}_{now}_{pid:d}.log',
                    logging_verbosity='debug',
                    logging_message_format='%(levelname)s %(asctime)s %(name)s: %(message)s | %(pathname)s.%(funcName)s:%(lineno)d'#,
                    #logging_datetime_format=config.logging_datetime_format
                  )


    fitall(systemname,systempath)
