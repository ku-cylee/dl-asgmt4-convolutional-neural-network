import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        # W: (cout, cin, wfil, hfil)
        # b: (1, cout, 1, 1)
        # x: (b, cin, win, hin)
        # y: (b, cout, wout, hout)

        b, cin, _, _ = x.shape
        cout, _, wfil, hfil = self.W.shape

        reshaped_W = self.W.reshape(cout, 1, -1)

        windows = np.squeeze(view_as_windows(x, (1, cin, wfil, hfil)), axis=(1, 4))
        _, wout, hout, _, _, _ = windows.shape
        reshaped_windows = windows.reshape(b, wout, hout, -1, 1)

        return np.swapaxes(np.squeeze(reshaped_W.dot(reshaped_windows), axis=(1, -1)), 0, 1) + self.b

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        cout, cin, wfil, hfil = self.W.shape

        # dLdW: (cout, cin, wfil, hfil)
        x_windows = np.squeeze(view_as_windows(x, (1, 1, wfil, hfil)), axis=(4, 5)) \
                      .transpose(1, 4, 5, 0, 2, 3) \
                      .reshape((cin, wfil, hfil, -1, 1))
        dLdy_flattened = np.swapaxes(dLdy, 0, 1).reshape(cout, 1, -1)
        dLdW = np.squeeze(dLdy_flattened.dot(x_windows), axis=(1, -1))

        # dLdx: (b, cin, win = wout + wfil - 1, hin = hout + hfil - 1)
        dLdy_padded = np.pad(dLdy, ((0, 0), (0, 0), (wfil - 1, wfil - 1), (hfil - 1, hfil - 1)), mode='constant')
        dLdy_windows = np.squeeze(view_as_windows(dLdy_padded, (1, cout, wfil, hfil)), axis=(1, 4))
        dLdy_flattened = dLdy_windows.reshape(dLdy_windows.shape[:3] + (1, -1))
        W_flipped = np.swapaxes(np.flip(self.W, axis=(2, 3)), 0, 1).reshape(cin, -1, 1)
        dLdx = np.squeeze(dLdy_flattened.dot(W_flipped), axis=(3, 5)).transpose(0, 3, 1, 2)

        # dLdb: (1, cout, 1, 1)
        dLdb = dLdy.sum(axis=3).sum(axis=2).sum(axis=0).reshape(self.b.shape)

        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        windows = view_as_windows(x,
                                  (1, 1, self.pool_size, self.pool_size),
                                  step=(1, 1, self.stride, self.stride))
        return np.max(windows, axis=(4, 5, 6, 7))

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        windows = view_as_windows(x,
                                  (1, 1, self.pool_size, self.pool_size),
                                  step=(1, 1, self.stride, self.stride))
        windowed_x = np.squeeze(windows, axis=(4, 5))

        y = self.forward(x)
        y_extended = y[:, :, :, :, None, None]
        _, _, wout, hout = y.shape

        dLdx = np.zeros_like(x, dtype=np.float64)
        for wout_idx in range(wout):
            for hout_idx in range(hout):
                window = windowed_x[:, :, wout_idx, hout_idx]
                max_value = y_extended[:, :, wout_idx, hout_idx]
                dLdy_value = dLdy[:, :, wout_idx, hout_idx, None, None]
                dLdx_submatrix = np.where(window == max_value, dLdy_value, 0)

                win_idx = wout_idx * self.stride
                hin_idx = hout_idx * self.stride
                dLdx[:, :, win_idx:win_idx + self.pool_size, hin_idx:hin_idx + self.pool_size] += dLdx_submatrix

        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')
