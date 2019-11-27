from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne
from parmesan.distributions import log_stdnormal, log_normal2
from parmesan.layers import NormalizingPlanarFlowLayer, ListIndexLayer


# config
test_energy_fun = 2
use_linear_nf0 = True#False          # use 0th linear NF to transform std gaussian to diagonal gaussian
use_annealed_loss = True        # use annealed free energy, see paper
nflows = 32
batch_size = 100
nparam_updates = 750000#500000
report_every = 1000
nepochs = nparam_updates // report_every # total num. of 'epochs'
lr = 1.0e-5
momentum = 0.9
iw_samples = 1  # number of samples for Monte Carlo approximation of expectation over q0(z)
                # note: because we aren't using any actual data in this case, this has the same effect as multiplying batch_size


class NormalizingLinearFlowLayer(lasagne.layers.Layer):
    """
    Normalizing flow layer with transform `f(z) = mu + sigma*z`.
    
    Ensures constraint `sigma > 0`, by reparameterizing `log sigma^2` as `log_var`.

    This flow transformation is not very powerful as a general transformation, 
    it is mainly intended for testing purposes.
    """
    def __init__(self, incoming,
                 mu=lasagne.init.Normal(),
                 log_var=lasagne.init.Normal(), **kwargs):
        super(NormalizingLinearFlowLayer, self).__init__(incoming, **kwargs)
        
        ndim = int(np.prod(self.input_shape[1:]))
        
        self.mu = self.add_param(mu, (ndim,), name="mu")
        self.log_var = self.add_param(log_var, (ndim,), name="log_var")
    
    def get_output_shape_for(self, input_shape):
        return input_shape
    
    def get_output_for(self, input, **kwargs):
        z = input
        sigma = T.exp(0.5*self.log_var)
        f_z = self.mu + sigma*z
        logdet_jacobian = T.sum(0.5*self.log_var)
        return [f_z, logdet_jacobian]

def U_z(z):
    """Test energy function U(z)."""
    z1 = z[:, 0]
    z2 = z[:, 1]

    if test_energy_fun == 1:
        return 0.5*((T.sqrt(z1**2 + z2**2) - 2)/0.4)**2 - T.log(T.exp(-0.5*((z1 - 2)/0.6)**2) + T.exp(-0.5*((z1 + 2)/0.6)**2))
    elif test_energy_fun == 2:
        w1 = T.sin((2.*np.pi*z1)/4.)
        return 0.5*((z2 - w1) / 0.4)**2
    elif test_energy_fun == 3:
        w1 = T.sin((2.*np.pi*z1)/4.)
        w2 = 3.*T.exp(-0.5*((z1 - 1)/0.6)**2)
        return -T.log(T.exp(-0.5*((z2 - w1)/0.35)**2) + T.exp(-0.5*((z2 - w1 + w2)/0.35)**2))
    elif test_energy_fun == 4:
        w1 = T.sin((2.*np.pi*z1)/4.)
        w3 = 3.*T.nnet.sigmoid((z1 - 1)/0.3)**4
        return -T.log(T.exp(-0.5*((z2 - w1)/0.4)**2) + T.exp(-0.5*((z2 - w1 + w3)/0.35)**2))
    else:
        raise ValueError('invalid `test_energy_fun`')

def evaluate_bivariate_pdf(p_z, range, npoints, z_sym=None):
    """Evaluate (possibly unnormalized) pdf over a meshgrid."""
    side = np.linspace(range[0], range[1], npoints)
    z1, z2 = np.meshgrid(side, side)
    z = np.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])

    if not z_sym:
        z_sym = T.matrix('z')
        p_z_ = p_z(z_sym)
    else:
        p_z_ = p_z

    p_z_fun = theano.function([z_sym], [p_z_], allow_input_downcast=True)
    
    return z1, z2, p_z_fun(z)[0].reshape((npoints, npoints))

def evaluate_bivariate_pdf_no_comp(p_z_fun, range, npoints):
    side = np.linspace(range[0], range[1], npoints)
    z1, z2 = np.meshgrid(side, side)
    z = np.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])
    out = p_z_fun(z)
    pK_z = out[0].reshape((npoints, npoints))
    zK = out[1]
    zK_1 = zK[:, 0].reshape((npoints, npoints))
    zK_2 = zK[:, 1].reshape((npoints, npoints))
    return zK_1, zK_2, pK_z

# main script
np.random.seed(1234)

z0 = T.matrix('z0')

l_in = lasagne.layers.InputLayer((None, 2), input_var=z0)

l_zk_list = [l_in]
l_nf_list = []
l_logdet_J_list = []

if use_linear_nf0:
    # also learn initial mapping form standard gaussian to gaussian with mean and diagonal covariance
    l_nf = NormalizingLinearFlowLayer(l_zk_list[-1], name='NF_0')
    l_zk = ListIndexLayer(l_nf, index=0)
    l_logdet_J = ListIndexLayer(l_nf, index=1)
    l_nf_list += [l_nf]
    l_zk_list += [l_zk]
    l_logdet_J_list += [l_logdet_J]

for k in xrange(nflows):
    # model parameter initialization
    # 1. current Parmesan defaults
    #u0 = lasagne.init.Normal() # = normal(std=1e-2)
    #w0 = lasagne.init.Normal() # = normal(std=1e-2)
    #b0 = lasagne.init.Constant(0.0)

    # 2. close to identity map, but not exactly 0 (normal(std=1e-3))
    u0 = lasagne.init.Normal(std=1e-3)
    w0 = lasagne.init.Normal(std=1e-3)
    b0 = lasagne.init.Normal(std=1e-3)
    
    # 3. close to identity map, but not exactly 0 (normal(std=1e-2))
    #u0 = lasagne.init.Normal(std=1e-2)
    #w0 = lasagne.init.Normal(std=1e-2)
    #b0 = lasagne.init.Normal(std=1e-2)
    
    # 4. trial-and-error (works well with test fun 1, annealing, q0(z) = std gaussian, 500k updates, rmsprop+momentum)
    #u0 = lasagne.init.Normal(mean=0.0, std=1.0)
    #w0 = lasagne.init.Uniform(range=(-1.0, 1.0))
    #b0 = lasagne.init.Constant(0.0)
    # ------------------------------
    l_nf = NormalizingPlanarFlowLayer(l_zk_list[-1], u0, w0, b0, name='NF_{:d}'.format(1+k))
    l_zk = ListIndexLayer(l_nf, index=0)
    l_logdet_J = ListIndexLayer(l_nf, index=1)
    l_nf_list += [l_nf]
    l_zk_list += [l_zk]
    l_logdet_J_list += [l_logdet_J]

if use_linear_nf0:
    train_out = lasagne.layers.get_output([l_zk_list[1], l_zk_list[-1]] + l_logdet_J_list, z0, deterministic=False)
    z1 = train_out[0] # z1 (= true z0) for plotting purposes
    zK = train_out[1]
    logdet_J_list = train_out[2:]
else:
    train_out = lasagne.layers.get_output([l_zk_list[-1]] + l_logdet_J_list, z0, deterministic=False)
    zK = train_out[0]
    logdet_J_list = train_out[1:]

# loss function
log_q0_z0 = log_stdnormal(z0).sum(axis=1)

sum_logdet_J = sum(logdet_J_list)
log_qK_zK = log_q0_z0 - sum_logdet_J

if use_annealed_loss:
    iteration = theano.shared(0)
    beta_t = T.minimum(1, 0.01 + iteration/10000)   # inverse temperature that goes from 0.01 to 1 after 10000 iterations
                                                    # XXX: an 'iteration' is a parameter update, right?
    kl = T.mean(log_qK_zK + beta_t*U_z(zK), axis=0)
else:
    kl = T.mean(log_qK_zK + U_z(zK), axis=0)
loss = kl

# updates
params = lasagne.layers.get_all_params([l_zk_list[-1]], trainable=True)

print('Trainable parameters:')
for p in params:
    print('  {}: {}'.format(p, p.get_value().shape if p.get_value().shape != () else 'scalar'))

grads = T.grad(loss, params)

updates_rmsprop = lasagne.updates.rmsprop(grads, params, learning_rate=lr)
updates = lasagne.updates.apply_momentum(updates_rmsprop, params, momentum=momentum)
#updates = lasagne.updates.adam(grads, params, learning_rate=0.0001) # XXX: adam learning rate does not seem to correspond exactly to RMSProp+momentum learning rate

if use_annealed_loss:
    updates[iteration] = iteration + 1

# compile
print('Compiling...')
train_model = theano.function([z0], [loss], updates=updates, allow_input_downcast=True)

# plot
def _plot(fn_png):
    fig = plt.figure()
    
    ax = plt.subplot(1, 5, 1, aspect='equal')
    mesh_z1, mesh_z2, phat_z = evaluate_bivariate_pdf(lambda z: T.exp(-U_z(z)), range=(-4, 4), npoints=200)
    plt.pcolormesh(mesh_z1, mesh_z2, phat_z)    # for plotting phat_z vs p_z doesn't matter much because plot is normalized
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$p(z)$')

    ax = plt.subplot(1, 5, 2, aspect='equal')
    if not use_linear_nf0:
        mesh_z1, mesh_z2, q0_z0_ = evaluate_bivariate_pdf(T.exp(log_q0_z0), range=(-4, 4), npoints=200, z_sym=z0)
        plt.pcolormesh(mesh_z1, mesh_z2, q0_z0_)
    else:
        log_q1_z1 = log_q0_z0 - logdet_J_list[0]
        q1_z1 = T.exp(log_q1_z1)
        eval_model = theano.function([z0], [q1_z1, z1], allow_input_downcast=True)

        mesh_z1, mesh_z2, q1_z1_ = evaluate_bivariate_pdf_no_comp(eval_model, range=(-4, 4), npoints=400) # use more points because z1, z2 meshgrid is nonlinearly transformed
        plt.pcolormesh(mesh_z1, mesh_z2, q1_z1_)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        cmap = matplotlib.cm.get_cmap(None)
        ax.set_facecolor(cmap(0.))

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$q_0(z)$')

    ax = plt.subplot(1, 5, 3, aspect='equal')

    qK_zK = T.exp(log_qK_zK)
    eval_model = theano.function([z0], [qK_zK, zK], allow_input_downcast=True)
    mesh_z1, mesh_z2, qK_zK_ = evaluate_bivariate_pdf_no_comp(eval_model, range=(-4, 4), npoints=400) # use more points because z1, z2 meshgrid is nonlinearly transformed
    ax.pcolormesh(mesh_z1, mesh_z2, qK_zK_)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    cmap = matplotlib.cm.get_cmap(None)
    ax.set_facecolor(cmap(0.))

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$q_K(z_K)$') # = q_K(f_1(f_2(..f_K(z))))

    def plot_hyper_plane(ax, w, b):
        """
        Planar flow contracts or expands the input density in the direction perpendicular 
        to the hyperplane w^Tz + b = 0.

        For the two-dimensional case we have
            [w1, w2]^T [z1, z2] + b = 0
            w1 z1 + w2 z2 + b = 0

            z2 = -(w1 z1 + b) / w2
                = -w1/w2 z1 - b/w2

            slope = -w1/w2, offset = -b/w2
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        z1 = np.array(xlim)

        z2 = -(w[0]*z1 + b)/w[1]
    
        z2 = z2 + b/w[1] # shift so line goes through origin (for visualization)
    
        #slope = -w[0]/w[1]
        #print(slope)
    
        p1 = [z1[0], z2[0]]
        p2 = [z1[1], z2[1]]
    
        #print(p1)
        #print(p2)
    
        ax.plot(z1, z2, '-r', linewidth=1.0)
        ax.plot(z1, z2, '--k', linewidth=1.0)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax = plt.subplot(1, 5, 4, aspect='equal')
    eval_model2 = theano.function([z0], [zK], allow_input_downcast=True)
    N = 1000000 # take many samples; but will still look a little 'spotty'
    z0_ = np.random.normal(size=(N, 2))
    zK_ = eval_model2(z0_)[0]
    ax.hist2d(zK_[:, 0], zK_[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$z_K \sim q_K(z)$')

    ax = plt.subplot(1, 5, 5, aspect='equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    for k in xrange(nflows):
        if hasattr(l_nf_list[k], 'w') and hasattr(l_nf_list[k], 'b'):
            plot_hyper_plane(ax, l_nf_list[k].w.get_value(), l_nf_list[k].b.get_value())
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$w^Tz + b = 0$')

    fig.tight_layout()

    #plt.show()
    plt.savefig(fn_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


# train
print('Training...')
losses = []
epoch = 0
assert nparam_updates % report_every == 0, 'should be integer multiple'
for k in xrange(nparam_updates):
    # before starting training (for checking initialization)
    if k == 0:
        # save plot
        _plot('epoch_{:04d}_init.png'.format(epoch))

        # display parameters
        for kk, p in enumerate(params):
            if use_linear_nf0 and kk < 2:
                if kk == 1:
                    print('  {}:\t{}'.format(p.name.replace('log_', ''), np.exp(p.get_value())))
                else:
                    print('  {}:\t{}'.format(p.name, p.get_value()))

    # generate z0
    z0_ = np.random.normal(size=(batch_size*iw_samples, 2))

    # train
    loss = train_model(z0_)[0]
    losses += [loss]

    # report
    if (1+k) % report_every == 0:
        avg_loss = np.mean(losses)
        print('{:d}/{:d}: loss = {:.6f}'.format(1+epoch, nepochs, avg_loss))

        if any(np.isnan(losses)):
            print('NaN loss, aborting...')
            for kk, loss in enumerate(losses):
                print('  Mini-batch {:04d}: {:.6f}'.format(1+kk, loss[()]))
                if np.isnan(loss[()]):
                    break
            break

        losses = []
        epoch += 1

        # save plot
        _plot('epoch_{:04d}_{:.5f}.png'.format(epoch, avg_loss))

        # display parameters
        for kk, p in enumerate(params):
            if use_linear_nf0 and kk < 2:
                if kk == 1:
                    print('  {}:\t{}'.format(p.name.replace('log_', ''), np.exp(p.get_value())))
                else:
                    print('  {}:\t{}'.format(p.name, p.get_value()))

print('Done :)')