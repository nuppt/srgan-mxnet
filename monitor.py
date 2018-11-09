import mxnet.ndarray as nd

# def plot_img(losses_log):
def plot_loss(sw, losses_log, global_step, epoch, i):
    message = '(epoch: %d, iters: %d) ' % (epoch, i)
    for key,value in losses_log.losses.items():
        if 'err' in key:
            loss = nd.concatenate(value,axis=0).mean().asscalar()
            sw.add_scalar('err', {key : loss}, global_step)
            message += '%s: %.6f ' % (key, loss)
    print(message)

def plot_img(sw, losses_log):
    sw.add_image(tag='lr_img', image=nd.clip(nd.concatenate([losses_log['lr_img'][0][0:4]]), 0, 1))
    sw.add_image(tag='hr_img', image=nd.clip(nd.concatenate([losses_log['hr_img'][0][0:4]]), 0, 1))
    sw.add_image(tag='hr_img_fake', image=nd.clip(nd.concatenate([losses_log['hr_img_fake'][0][0:4]]), 0, 1))
