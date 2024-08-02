import os
import json

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from cfg_train import get_cfg
from agent.ppo import *
from environment.data import DataGenerator
from environment.env import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate(val_dir):
    val_paths = os.listdir(val_dir)
    with torch.no_grad():
        average_delay_lst, average_move_lst, average_priority_ratio_lst = [], [], []
        for path in val_paths:
            test_env = QuayScheduling(data_src=val_dir + path, look_ahead=look_ahead,
                                      w_delay=w_delay, w_move=w_move, w_priority=w_priority,
                                      algorithm="RL", state_encoding=encoding, restriction=restriction,
                                      record_events=False, device=device)

            state, mask, current_ops, added_info = test_env.reset()
            done = False

            while not done:
                action, _, _ = agent.get_action(state, mask, current_ops, added_info)
                next_state, reward, done, next_mask, next_current_ops, next_added_info = test_env.step(action)

                state = next_state
                mask = next_mask
                current_ops = next_current_ops
                added_info = next_added_info

                if done:
                    # log = test_env.get_logs()
                    break

            delay = sum(test_env.monitor.delay.values()) / len(test_env.monitor.delay.values())
            move = sum(test_env.monitor.move.values()) / len(test_env.monitor.move.values())
            priority_ratio = sum(test_env.monitor.priority_ratio.values()) / len(test_env.monitor.priority_ratio.values())

            average_delay_lst.append(delay)
            average_move_lst.append(move)
            average_priority_ratio_lst.append(priority_ratio)

        average_delay = sum(average_delay_lst) / len(average_delay_lst)
        average_move = sum(average_move_lst) / len(average_move_lst)
        average_priority_ratio = sum(average_priority_ratio_lst) / len(average_priority_ratio_lst)

        return average_delay, average_move, average_priority_ratio


if __name__ == "__main__":
    date = datetime.now().strftime('%m%d_%H_%M')
    cfg = get_cfg()
    if cfg.vessl == 1:
        import vessl
        vessl.init(organization="snu-eng-dgx", project="quay", hp=cfg)

    n_episode = cfg.n_episode
    load_model = cfg.load_model

    n_ships = cfg.n_ships
    data_path = cfg.data_path

    use_gnn = bool(cfg.use_gnn)
    use_added_info = bool(cfg.use_added_info)
    encoding = cfg.encoding
    restriction = bool(cfg.restriction)
    look_ahead = cfg.look_ahead
    embed_dim = cfg.embed_dim
    num_heads = cfg.num_heads
    num_HGT_layers = cfg.num_HGT_layers
    num_actor_layers = cfg.num_actor_layers
    num_critic_layers = cfg.num_critic_layers
    lr = cfg.lr
    lr_decay = cfg.lr_decay
    lr_step = cfg.lr_step
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon
    P_coeff = cfg.P_coeff
    V_coeff = cfg.V_coeff
    E_coeff = cfg.E_coeff

    w_delay = cfg.w_delay
    w_move = cfg.w_move
    w_priority = cfg.w_priority

    eval_every = cfg.eval_every
    save_every = cfg.save_every
    new_instance_every = cfg.new_instance_every

    val_dir = cfg.val_dir

    if cfg.vessl == 1:
        model_dir = '/output/train/' + date + '/model/'
    elif cfg.vessl == 0:
        model_dir = './output/train/' + date + '/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if cfg.vessl == 1:
        log_dir = '/output/train/' + date + '/log/'
    elif cfg.vessl == 0:
        log_dir = './output/train/' + date + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # simulation_dir = '../output/train/simulation/'
    # if not os.path.exists(simulation_dir):
    #    os.makedirs(simulation_dir)

    with open(log_dir + "parameters.json", 'w') as f:
        json.dump(vars(cfg), f, indent=4)

    data_generator = DataGenerator(n_ships, data_path)
    env = QuayScheduling(data_generator,
                         look_ahead=look_ahead, w_delay=w_delay, w_move=w_move, w_priority=w_priority,
                         algorithm="RL", state_encoding=encoding, restriction=restriction,
                         record_events=False, device=device)
    agent = Agent(env.meta_data, env.state_size, env.num_nodes, embed_dim, num_heads,
                  num_HGT_layers, num_actor_layers, num_critic_layers, lr, lr_decay, lr_step,
                  gamma, lmbda, eps_clip, K_epoch, P_coeff, V_coeff, E_coeff,
                  use_gnn=use_gnn, use_added_info=use_added_info, device=device)
    if cfg.vessl == 0:
        writer = SummaryWriter(log_dir)

    if bool(load_model):
        checkpoint = torch.load(cfg.model_path)
        start_episode = checkpoint['episode'] + 1
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss, lr\n')
    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('episode, average_delay, move_ratio, priority_ratio\n')

    for e in range(start_episode, n_episode + 1):
        if cfg.vessl == 1:
            vessl.log(payload={"Train/learnig_rate": agent.scheduler.get_last_lr()[0]}, step=e)
        elif cfg.vessl == 0:
            writer.add_scalar("Training/Learning Rate", agent.scheduler.get_last_lr()[0], e)

        n = 0
        r_epi = 0.0
        avg_loss = 0.0
        done = False

        state, mask, current_ops, added_info = env.reset()

        while not done:
            for t in range(T_horizon):
                action, action_logprob, state_value = agent.get_action(state, mask, current_ops, added_info)
                next_state, reward, done, next_mask, next_current_ops, next_added_info = env.step(action)

                agent.put_data((state, added_info, action, reward, next_state, next_added_info,
                                action_logprob, state_value, mask, current_ops, done))
                state = next_state
                mask = next_mask
                current_ops = next_current_ops
                added_info = next_added_info

                r_epi += reward

                if done:
                    break

            n += 1
            avg_loss += agent.train()
        agent.scheduler.step()

        print("episode: %d | reward: %.4f | loss: %.4f" % (e, r_epi, avg_loss / n))
        with open(log_dir + "train_log.csv", 'a') as f:
            f.write('%d, %1.4f, %1.4f, %f\n' % (e, r_epi, avg_loss, agent.scheduler.get_last_lr()[0]))

        if cfg.vessl == 1:
            vessl.log(payload={"Train/reward": r_epi, "Train/loss": avg_loss / n}, step=e)
        elif cfg.vessl == 0:
            writer.add_scalar("Training/Reward", r_epi, e)
            writer.add_scalar("Training/Loss", avg_loss / n, e)

        if e == start_episode or e % eval_every == 0:
            average_delay, average_move, average_priority_ratio = evaluate(val_dir)

            with open(log_dir + "validation_log.csv", 'a') as f:
                f.write('%d,%1.4f, %1.4f, %1.4f\n' % (e, average_delay, average_move, average_priority_ratio))

            if cfg.vessl == 1:
                vessl.log(payload={"Perf/average_delay": average_delay,
                                   "Perf/move_ratio": average_move,
                                   "Perf/priority_ratio": average_priority_ratio}, step=e)
            elif cfg.vessl == 0:
                writer.add_scalar("Validation/Average Delay", average_delay, e)
                writer.add_scalar("Validation/Move Ratio", average_move, e)
                writer.add_scalar("Validation/Priority Ratio", average_priority_ratio, e)

        if e % save_every == 0:
            agent.save_network(e, model_dir)

        if e % new_instance_every == 0:
            env.generate_new_instance()

    if cfg.vessl == 0:
        writer.close()

