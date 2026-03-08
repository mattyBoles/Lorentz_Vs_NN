import torch
import numpy as np
import matplotlib.pyplot as plt


def rollout(model, ic, n_steps, device):
    x0 = torch.tensor(ic, dtype=torch.float32).to(device)
    trajectory = []
    trajectory.append(x0.cpu().numpy())
    model.eval()
    with torch.inference_mode():
        for _ in range(n_steps):
            x0 = model(x0)
            trajectory.append(x0.cpu().numpy())
    return np.vstack(trajectory)


def denormalise(trajectory, mean, std):
    return trajectory * std.numpy() + mean.numpy()


def plot_phase_portrait(traj1, traj2, save_path='phase_portrait.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], linewidth=0.5, color='blue', label='IC 1')
    ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], linewidth=0.5, color='red',  label='IC 2')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.savefig(save_path)
    plt.close()


def plot_sync(traj1, traj2, h, save_path='sync.png'):
    t = np.arange(0, len(traj1) * h, h)
    fig, ax = plt.subplots()
    ax.plot(t, traj1[:,0], color='blue', label='IC 1')
    ax.plot(t, traj2[:,0], color='red',  label='IC 2')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()
    plt.savefig(save_path)
    plt.close()


def plot_loss(history, save_path='loss.png'):
    fig, ax = plt.subplots()
    history['train'] = [epoch_loss.to('cpu') for epoch_loss in history['train']]
    history['val'] = [epoch_loss.to('cpu') for epoch_loss in history['val']]
    ax.plot(history['train'], label='train')
    ax.plot(history['val'],   label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    plt.savefig(save_path)
    plt.close()

def evaluate(model, history, config, mean, std):
    device = config['device']
    n_steps = config['n_rollout_steps']
    h = config['h']

    ic1 =  [(config['ic1'][0] - mean[0])/std[0], (config['ic1'][1] - mean[1])/std[1], (config['ic1'][2] - mean[2])/std[2]]
    ic2 =  [(config['ic2'][0] - mean[0])/std[0], (config['ic2'][1] - mean[1])/std[1], (config['ic2'][2] - mean[2])/std[2]]

    # Rollouts
    traj1 = rollout(model, ic1, n_steps, device)
    traj2 = rollout(model, ic2, n_steps, device)

    # Denormalise
    traj1 = denormalise(traj1, mean, std)
    traj2 = denormalise(traj2, mean, std)

    # Plots
    plot_phase_portrait(traj1, traj2)
    plot_sync(traj1, traj2, h)
    plot_loss(history)

    print('Evaluation plots saved.')