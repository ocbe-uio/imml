import os.path
import torch
from pandarallel import pandarallel
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imvc.decomposition import DFMF, MOFA, DeepMF
from imvc.preprocessing import MultiViewTransformer, ConcatenateViews, NormalizerNaN
from imvc.cluster import NEMO, DAIMC, PIMVC, SIMCADC, OSLFIMVC, MSNE, MKKMIK, LFIMVC, EEIMVC, SUMO, OPIMC, OMVC, MONET, \
    IMSR

from settings import INCOMPLETE_RESULTS_PATH, INCOMPLETE_SUBRESULTS_PATH, INCOMPLETE_LOGS_PATH, INCOMPLETE_ERRORS_PATH, \
    TIME_RESULTS_PATH, DATASET_TABLE_PATH, amputation_mechanisms, probs, \
    imputation, runs_per_alg, INCOMPLETE_RESULTS_FILE, INCOMPLETE_LOGS_FILE, INCOMPLETE_ERRORS_FILE, \
    INCOMPLETE_SUBRESULTS_FOLDER
from src.commons import CommonOperations

args = CommonOperations.get_args()

if not args.save_results:
    results_folder = 'test'
    INCOMPLETE_RESULTS_PATH = os.path.join(results_folder, INCOMPLETE_RESULTS_FILE)
    INCOMPLETE_LOGS_PATH = os.path.join(results_folder, INCOMPLETE_LOGS_FILE)
    INCOMPLETE_ERRORS_PATH = os.path.join(results_folder, INCOMPLETE_ERRORS_FILE)
    INCOMPLETE_SUBRESULTS_PATH = os.path.join(results_folder, INCOMPLETE_SUBRESULTS_FOLDER)

if args.n_jobs > 1:
    pandarallel.initialize(nb_workers= args.n_jobs)

algorithms = {
    "DAIMC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   DAIMC()), "params": {}},
    "EEIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    EEIMVC()), "params": {}},
    "IMSR": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   IMSR()), "params": {}},
    "LFIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    LFIMVC()), "params": {}},
    "MKKMIK": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                    MKKMIK()), "params": {}},
    # "MONET": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                                 MONET()), "params": {}},
    # "MSNE": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                               MSNE()), "params": {}},
    "OMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  OMVC()), "params": {}},
    "OPIMC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                     OPIMC()), "params": {}},
    "OSLFIMVC": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                      OSLFIMVC()), "params": {}},
    "PIMVC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                   PIMVC()), "params": {}},
    "SIMCADC": {"alg": make_pipeline(MultiViewTransformer(NormalizerNaN().set_output(transform="pandas")),
                                     SIMCADC()), "params": {}},
    # "SUMO": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
    #                                   SUMO()), "params": {}},
    "DeepMF": {"alg": make_pipeline(ConcatenateViews(), StandardScaler(),
                                    FunctionTransformer(lambda x: torch.from_numpy(x).float().cuda().t()),
                                    DeepMF(), FunctionTransformer(lambda x: x.cpu().detach().numpy()),
                                    StandardScaler(), KMeans()),
                 "params": {}},
    "DFMF": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  DFMF().set_output(transform="pandas"),
                                  StandardScaler().set_output(transform="pandas"), KMeans()),
             "params": {}},
    "MOFA": {"alg": make_pipeline(MultiViewTransformer(StandardScaler().set_output(transform="pandas")),
                                  MOFA().set_output(transform="pandas"),
                                  ConcatenateViews(), StandardScaler().set_output(transform="pandas"), KMeans()),
             "params": {}},
}
incomplete_algorithms = True
CommonOperations.run_script(dataset_table_path=DATASET_TABLE_PATH, algorithms=algorithms, probs=probs,
                            amputation_mechanisms=amputation_mechanisms, imputation=imputation,
                            runs_per_alg=runs_per_alg, args=args, subresults_path=INCOMPLETE_SUBRESULTS_PATH,
                            logs_file=INCOMPLETE_LOGS_PATH, error_file=INCOMPLETE_ERRORS_PATH,
                            results_path=INCOMPLETE_RESULTS_PATH, time_results_path=TIME_RESULTS_PATH,
                            incomplete_algorithms=incomplete_algorithms)
