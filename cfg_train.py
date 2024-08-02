import argparse


def get_cfg():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--vessl", type=int, default=1, help="whether to use vessl (0: False, 1:True)")

    parser.add_argument("--n_episode", type=int, default=10000, help="number of episodes")
    parser.add_argument("--load_model", type=int, default=0, help="whether to load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--n_ships", type=int, default=80, help="number of ships in data")
    parser.add_argument("--data_path", type=str, default=None, help="input data path")

    parser.add_argument("--use_gnn", type=int, default=1, help="whether to use gnn")
    parser.add_argument("--use_added_info", type=int, default=1, help="whether to use additional information")
    parser.add_argument("--encoding", type=str, default="DG", help="state encoding method")
    parser.add_argument("--restriction", type=int, default=0, help="whether to use restricted action space (0: False, 1:True)")
    parser.add_argument("--look_ahead", type=int, default=3, help="number of operations included in states")
    parser.add_argument("--embed_dim", type=int, default=128, help="node embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="multi-head attention in HGT layers")
    parser.add_argument("--num_HGT_layers", type=int, default=2, help="number of HGT layers")
    parser.add_argument("--num_actor_layers", type=int, default=2, help="number of actor layers")
    parser.add_argument("--num_critic_layers", type=int, default=2, help="number of critic layers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="learning rate decay ratio")
    parser.add_argument("--lr_step", type=int, default=2000, help="step size to reduce learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping paramter")
    parser.add_argument("--K_epoch", type=int, default=5, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=10, help="the number of steps to obtain samples")
    parser.add_argument("--P_coeff", type=float, default=1, help="coefficient for policy loss")
    parser.add_argument("--V_coeff", type=float, default=0.5, help="coefficient for value loss")
    parser.add_argument("--E_coeff", type=float, default=0.01, help="coefficient for entropy loss")

    parser.add_argument("--w_delay", type=float, default=0.0, help="weight for minimizing delays")
    parser.add_argument("--w_move", type=float, default=1.0, help="weight for minimizing the number of ship movements")
    parser.add_argument("--w_priority", type=float, default=1.0, help="weight for maximizing the efficiency")

    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x episodes")
    parser.add_argument("--new_instance_every", type=int, default=10, help="Generate new scenarios every x episodes")

    parser.add_argument("--val_dir", type=str, default=None, help="directory where the validation data are stored")

    return parser.parse_args()