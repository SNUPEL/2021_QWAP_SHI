import argparse


def get_cfg():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--vessl", type=int, default=1, help="whether to use vessl (0: False, 1:True)")

    parser.add_argument("--model_path", type=str, default=None, help="model file path")
    parser.add_argument("--param_path", type=str, default=None, help="hyper-parameter file path")
    parser.add_argument("--data_dir", type=str, default=None, help="test data path")
    parser.add_argument("--res_dir", type=str, default=None, help="test result file path")
    parser.add_argument("--sim_dir", type=str, default=None, help="simulation log file path")

    parser.add_argument("--algorithm", type=str, default="ALL", help="test algorithm")
    parser.add_argument("--use_gnn", type=int, default=1, help="whether to use gnn")
    parser.add_argument("--use_added_info", type=int, default=1, help="whether to use additional information")
    parser.add_argument("--encoding", type=str, default="BG", help="state encoding method")
    parser.add_argument("--restriction", type=int, default=0, help="whether to use restricted action space (0: False, 1:True)")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    return parser.parse_args()