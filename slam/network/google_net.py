
from kaffe.tensorflow.network import Network
from tensorflow.python.training.adam import AdamOptimizer

import tensorflow as tf


class GoogleNet(Network):
    
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, name='conv1_7x7_s2')
             .max_pool(3, 3, 2, 2, name='pool1_3x3_s2')
             .lrn(2, 2e-05, 0.75, name='pool1_norm1')
             .conv(1, 1, 64, 1, 1, name='conv2_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='conv2_3x3')
             .lrn(2, 2e-05, 0.75, name='conv2_norm2')
             .max_pool(3, 3, 2, 2, name='pool2_3x3_s2')
             .conv(1, 1, 64, 1, 1, name='inception_3a_1x1'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 96, 1, 1, name='inception_3a_3x3_reduce')
             .conv(3, 3, 128, 1, 1, name='inception_3a_3x3'))

        (self.feed('pool2_3x3_s2')
             .conv(1, 1, 16, 1, 1, name='inception_3a_5x5_reduce')
             .conv(5, 5, 32, 1, 1, name='inception_3a_5x5'))

        (self.feed('pool2_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_3a_pool')
             .conv(1, 1, 32, 1, 1, name='inception_3a_pool_proj'))

        (self.feed('inception_3a_1x1',
                   'inception_3a_3x3',
                   'inception_3a_5x5',
                   'inception_3a_pool_proj')
             .concat(3, name='inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_1x1'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 128, 1, 1, name='inception_3b_3x3_reduce')
             .conv(3, 3, 192, 1, 1, name='inception_3b_3x3'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 32, 1, 1, name='inception_3b_5x5_reduce')
             .conv(5, 5, 96, 1, 1, name='inception_3b_5x5'))

        (self.feed('inception_3a_output')
             .max_pool(3, 3, 1, 1, name='inception_3b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_3b_pool_proj'))

        (self.feed('inception_3b_1x1',
                   'inception_3b_3x3',
                   'inception_3b_5x5',
                   'inception_3b_pool_proj')
             .concat(3, name='inception_3b_output')
             .max_pool(3, 3, 2, 2, name='pool3_3x3_s2')
             .conv(1, 1, 192, 1, 1, name='inception_4a_1x1'))

        (self.feed('pool3_3x3_s2')
             .conv(1, 1, 96, 1, 1, name='inception_4a_3x3_reduce')
             .conv(3, 3, 208, 1, 1, name='inception_4a_3x3'))

        (self.feed('pool3_3x3_s2')
             .conv(1, 1, 16, 1, 1, name='inception_4a_5x5_reduce')
             .conv(5, 5, 48, 1, 1, name='inception_4a_5x5'))

        (self.feed('pool3_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_4a_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4a_pool_proj'))

        (self.feed('inception_4a_1x1',
                   'inception_4a_3x3',
                   'inception_4a_5x5',
                   'inception_4a_pool_proj')
             .concat(3, name='inception_4a_output')
             .conv(1, 1, 160, 1, 1, name='inception_4b_1x1'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 112, 1, 1, name='inception_4b_3x3_reduce')
             .conv(3, 3, 224, 1, 1, name='inception_4b_3x3'))

        (self.feed('inception_4a_output')
             .conv(1, 1, 24, 1, 1, name='inception_4b_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4b_5x5'))

        (self.feed('inception_4a_output')
             .max_pool(3, 3, 1, 1, name='inception_4b_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4b_pool_proj'))
        
        (self.feed('inception_4b_1x1',
                   'inception_4b_3x3',
                   'inception_4b_5x5',
                   'inception_4b_pool_proj')
             .concat(3, name='inception_4b_output')
             .conv(1, 1, 128, 1, 1, name='inception_4c_1x1'))

        (self.feed('inception_4b_output')
             .conv(1, 1, 128, 1, 1, name='inception_4c_3x3_reduce')
             .conv(3, 3, 256, 1, 1, name='inception_4c_3x3'))

        (self.feed('inception_4b_output')
             .conv(1, 1, 24, 1, 1, name='inception_4c_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4c_5x5'))

        (self.feed('inception_4b_output')
             .max_pool(3, 3, 1, 1, name='inception_4c_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4c_pool_proj'))

        # # changes ...
        (self.feed('inception_4b_output')
             .avg_pool(5, 5, 3, 3, name='inception_4b_avg_pool')
             .conv(1, 1, 128, 1, 1, name='inception_4b_conv1x1')
             .fc(1024, name='inception_4b_fc1')
             .fc(1024, name='inception_4b_fc2')
             .fc(7, name='output1', relu=False))
        
        (self.feed('inception_4c_1x1',
                   'inception_4c_3x3',
                   'inception_4c_5x5',
                   'inception_4c_pool_proj')
             .concat(3, name='inception_4c_output')
             .conv(1, 1, 112, 1, 1, name='inception_4d_1x1'))

        (self.feed('inception_4c_output')
             .conv(1, 1, 144, 1, 1, name='inception_4d_3x3_reduce')
             .conv(3, 3, 288, 1, 1, name='inception_4d_3x3'))

        (self.feed('inception_4c_output')
             .conv(1, 1, 32, 1, 1, name='inception_4d_5x5_reduce')
             .conv(5, 5, 64, 1, 1, name='inception_4d_5x5'))

        (self.feed('inception_4c_output')
             .max_pool(3, 3, 1, 1, name='inception_4d_pool')
             .conv(1, 1, 64, 1, 1, name='inception_4d_pool_proj'))

        (self.feed('inception_4d_1x1',
                   'inception_4d_3x3',
                   'inception_4d_5x5',
                   'inception_4d_pool_proj')
             .concat(3, name='inception_4d_output')
             .conv(1, 1, 256, 1, 1, name='inception_4e_1x1'))

        (self.feed('inception_4d_output')
             .conv(1, 1, 160, 1, 1, name='inception_4e_3x3_reduce')
             .conv(3, 3, 320, 1, 1, name='inception_4e_3x3'))

        (self.feed('inception_4d_output')
             .conv(1, 1, 32, 1, 1, name='inception_4e_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_4e_5x5'))

        (self.feed('inception_4d_output')
             .max_pool(3, 3, 1, 1, name='inception_4e_pool')
             .conv(1, 1, 128, 1, 1, name='inception_4e_pool_proj'))

        # # output 2 
        (self.feed('inception_4d_output')
             .avg_pool(5, 5, 3, 3, name='inception_4d_avg_pool')
             .conv(1, 1, 128, 1, 1, name='inception_4d_conv1x1')
             .fc(1024, name='inception_4d_fc1')
             .fc(1024, name='inception_4d_fc2')
             .fc(7, name='output2', relu=False))
        
        (self.feed('inception_4e_1x1',
                   'inception_4e_3x3',
                   'inception_4e_5x5',
                   'inception_4e_pool_proj')
             .concat(3, name='inception_4e_output')
             .max_pool(3, 3, 2, 2, name='pool4_3x3_s2')
             .conv(1, 1, 256, 1, 1, name='inception_5a_1x1'))

        (self.feed('pool4_3x3_s2')
             .conv(1, 1, 160, 1, 1, name='inception_5a_3x3_reduce')
             .conv(3, 3, 320, 1, 1, name='inception_5a_3x3'))

        (self.feed('pool4_3x3_s2')
             .conv(1, 1, 32, 1, 1, name='inception_5a_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_5a_5x5'))

        (self.feed('pool4_3x3_s2')
             .max_pool(3, 3, 1, 1, name='inception_5a_pool')
             .conv(1, 1, 128, 1, 1, name='inception_5a_pool_proj'))

        (self.feed('inception_5a_1x1',
                   'inception_5a_3x3',
                   'inception_5a_5x5',
                   'inception_5a_pool_proj')
             .concat(3, name='inception_5a_output')
             .conv(1, 1, 384, 1, 1, name='inception_5b_1x1'))

        (self.feed('inception_5a_output')
             .conv(1, 1, 192, 1, 1, name='inception_5b_3x3_reduce')
             .conv(3, 3, 384, 1, 1, name='inception_5b_3x3'))

        (self.feed('inception_5a_output')
             .conv(1, 1, 48, 1, 1, name='inception_5b_5x5_reduce')
             .conv(5, 5, 128, 1, 1, name='inception_5b_5x5'))

        (self.feed('inception_5a_output')
             .max_pool(3, 3, 1, 1, name='inception_5b_pool')
             .conv(1, 1, 128, 1, 1, name='inception_5b_pool_proj'))

        (self.feed('inception_5b_1x1',
                   'inception_5b_3x3',
                   'inception_5b_5x5',
                   'inception_5b_pool_proj')
             .concat(3, name='inception_5b_output')
             .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5_7x7_s1')
             .fc(1024, relu=False, name='inception_5b_fc1')
             .fc(1024, relu=False, name='inception_5b_fc2')
             .fc(7, relu=False, name='output3'))
    
    def add_loss(self, groundtruth):
        output1, output2, output3 = self.layers['output1'], self.layers['output2'], self.layers['output3']
        loss1, loss2, loss3 = self.get_loss(output1, groundtruth), self.get_loss(output2, groundtruth), self.get_loss(output3, groundtruth)
        tf.scalar_summary('loss1', loss1)
        tf.scalar_summary('loss2', loss2)
        tf.scalar_summary('loss3', loss3)
        self.total_loss = 0.3 * loss1 + 0.3 * loss2 + loss3
        tf.scalar_summary('total_loss', self.total_loss)
        return self.total_loss
        
    def get_loss(self, output, groundtruth):
        beta = 1100
        x = groundtruth[:, :3]
        xp = output[:, :3]
        q = groundtruth[:, 3:]
        qp = output[:, 3:]
        loss = tf.reduce_sum((xp - x) ** 2) + beta * tf.reduce_sum((qp - q) ** 2)
        loss = tf.Print(loss, [output, groundtruth, loss], 'output, groundtruth and loss:', summarize=20)
        return loss
        
    def add_optimizer(self):
        global_step = tf.Variable(0, trainable=False)
        
        learning_rate = tf.train.exponential_decay(0.00001, global_step, 10,
                                   0.1, staircase=True)
        
        optimizer = AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.total_loss)
        
        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        return apply_gradient_op


if __name__ == '__main__':
    
    network_input = tf.placeholder(tf.float32, [10, 224, 224, 4], 'network_input')
    google_net = GoogleNet({'data':network_input})
    google_net.add_loss()
    google_net.add_optimizer()
    
