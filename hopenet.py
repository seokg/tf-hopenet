import tensorflow as tf
import os, argparse
from tensorflow.contrib.slim import nets
import datasets
from PIL import Image
import cv2
import numpy as np
from math import cos, sin

slim = tf.contrib.slim

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=10, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.00001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='./300W_LP', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='./300W_LP_filename_filtered.txt', type=str)
    parser.add_argument('--pretrained_path', dest='pretrained_path', help='Path to put the pretrained model,like resetnet50.', 
          default = './resnet50/', type=str)
    parser.add_argument('--ce', dest='ce', help='ce loss coefficient.',
          default=1.0, type=float)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=1.0, type=float)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', help='Path to save the model.',
          default='./ckpt', type=str)
    parser.add_argument('--image_size',dest = 'image_size', help = 'input image size', default = 224,type=int)
    parser.add_argument('--log_dir', dest='log_dir', help='log.',default='./output/log', type=str)

    args = parser.parse_args()
    print(args)
    return args


class hopenet():

    def __init__(self, args):
        self.args = args
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu_id)
        self.args.log_dir = self.args.log_dir + '-' + str(self.args.gpu_id)
        self.args.checkpoint_dir = self.args.checkpoint_dir + '-' + str(self.args.gpu_id)

        mae_yaw = 0
        mae_pitch = 0
        mae_roll = 0
        
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='image_batch')
        self.is_training = tf.placeholder(tf.bool,name='is_training')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape = [], name = 'keep_prob')

        self.num_bins = 66

        # init session
        self.config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)


    def build(self):
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net,endpoints = nets.resnet_v1.resnet_v1_50(self.images, num_classes=None, is_training=self.is_training)
        
        with tf.variable_scope('Logits'):
            net = tf.squeeze(net,axis=[1,2])
            net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training, scope='dropout')
            self.yaw   = slim.fully_connected(net, num_outputs=self.num_bins, activation_fn=None, scope='fc_yaw')
            self.pitch = slim.fully_connected(net, num_outputs=self.num_bins, activation_fn=None, scope='fc_pitch')
            self.roll  = slim.fully_connected(net, num_outputs=self.num_bins, activation_fn=None, scope='fc_roll')
            
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = tf.convert_to_tensor(idx_tensor,tf.float32)
        
        self.yaw_predicted   = tf.reduce_sum(tf.nn.softmax(self.yaw)   * idx_tensor, 1) * 3 - 99
        self.pitch_predicted = tf.reduce_sum(tf.nn.softmax(self.pitch) * idx_tensor, 1) * 3 - 99
        self.roll_predicted  = tf.reduce_sum(tf.nn.softmax(self.roll)  * idx_tensor, 1) * 3 - 99
    

    def train(self):
        self.labels = tf.placeholder(dtype=tf.int32, shape = [None,3], name = 'cls_label')
        self.cont_labels = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='cont_labels')
        
        # Binned labels
        self.label_yaw = self.labels[:,0]
        self.label_pitch = self.labels[:,1]
        self.label_roll = self.labels[:,2]
        
        # Continuous labels
        self.label_yaw_cont = self.cont_labels[:,0]
        self.label_pitch_cont = self.cont_labels[:,1]
        self.label_roll_cont = self.cont_labels[:,2]

        # Cross entropy loss
        loss_yaw   = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(self.label_yaw, tf.int64), logits=self.yaw)
        loss_pitch = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(self.label_pitch,tf.int64),logits=self.pitch)
        loss_roll  = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(self.label_roll,tf.int64),logits=self.roll)
        
        self.loss_yaw_ce   = tf.reduce_mean(loss_yaw)
        self.loss_pitch_ce = tf.reduce_mean(loss_pitch)
        self.loss_roll_ce  = tf.reduce_mean(loss_roll)
        
        # MSE loss
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = tf.convert_to_tensor(idx_tensor,dtype=tf.float32)
        
        self.loss_reg_yaw   = tf.reduce_mean(tf.square(self.yaw_predicted   - self.label_yaw_cont))
        self.loss_reg_pitch = tf.reduce_mean(tf.square(self.pitch_predicted - self.label_pitch_cont))
        self.loss_reg_roll  = tf.reduce_mean(tf.square(self.roll_predicted  - self.label_roll_cont))

        # Total loss
        self.loss_yaw   = self.args.ce * self.loss_yaw_ce   + self.args.alpha * self.loss_reg_yaw
        self.loss_pitch = self.args.ce * self.loss_pitch_ce + self.args.alpha * self.loss_reg_pitch
        self.loss_roll  = self.args.ce * self.loss_roll_ce  + self.args.alpha * self.loss_reg_roll
        self.loss_yaw   = self.args.ce * self.loss_yaw_ce   + self.args.alpha * self.loss_reg_yaw
        self.loss_pitch = self.args.ce * self.loss_pitch_ce + self.args.alpha * self.loss_reg_pitch
        self.loss_roll  = self.args.ce * self.loss_roll_ce  + self.args.alpha * self.loss_reg_roll
        
        self.loss_all = self.loss_yaw + self.loss_pitch + self.loss_roll
        
        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        rate = tf.train.exponential_decay(0.00001, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
        train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(self.loss_all, global_step=global_step)

        tf.summary.scalar('loss_mse_yaw', self.loss_reg_yaw)
        tf.summary.scalar('loss_mse_pitch', self.loss_reg_pitch)
        tf.summary.scalar('loss_mse_roll', self.loss_reg_roll)
        tf.summary.scalar('loss_ce_yaw', self.loss_yaw_ce)
        tf.summary.scalar('loss_ce_pitch', self.loss_pitch_ce)
        tf.summary.scalar('loss_ce_roll', self.loss_roll_ce)
        tf.summary.scalar('loss_yaw', self.loss_yaw)
        tf.summary.scalar('loss_pitch', self.loss_pitch)
        tf.summary.scalar('loss_roll', self.loss_roll)
        tf.summary.scalar('loss_all', self.loss_all)
        merged_summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        self.sess.run(init)

        if not os.path.exists(self.args.log_dir):
             os.makedirs(self.args.log_dir)
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        
        ckpt = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if (ckpt):
            tf.logging.info('restore the trained model')
            saver = tf.train.Saver(max_to_keep=5)
            saver.restore(self.sess,ckpt)
        else:
            print('[ ] load resnet model ....')
            tf.logging.info('load the pre-trained model')
            checkpoint_exclude_scopes = 'Logits'
            #exclusions = None
            
            if checkpoint_exclude_scopes:
                exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
                
            variables_to_restore = []
            for var in slim.get_model_variables():
                print(var)
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                    else:
                        variables_to_restore.append(var)
            
            saver_restore = tf.train.Saver(variables_to_restore)
            saver = tf.train.Saver(max_to_keep=5)
            saver_restore.restore(self.sess, os.path.join(self.args.pretrained_path,'resnet_v1_50.ckpt'))
            print('[*] finished loading resnet model ....')
        
        train_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        dataset = datasets.Pose_300W_LP(data_dir='./300W_LP', filename_path=self.args.filename_list,
                                        batch_size=self.args.batch_size,
                                        image_size=self.args.image_size)
        
        for epoch in range(self.args.num_epochs):
            for i in range(dataset.length//self.args.batch_size):
                batch_images, batch_labels, batch_cont_labels = dataset.get()
                train_dict = {self.images: batch_images, 
                              self.labels: batch_labels,
                              self.is_training: True,
                              self.keep_prob: 0.5,
                              self.cont_labels: batch_cont_labels}
                _, loss, yaw_loss, pitch_loss, roll_loss, train_summary, step = self.sess.run([train_op,
                        self.loss_all, self.loss_yaw, self.loss_pitch, self.loss_roll, merged_summary_op, global_step],feed_dict = train_dict)
                
                
                train_writer.add_summary(train_summary,step)
                
                # inference of predicted value
                if step % 100==0:
                    yaw, pitch, roll = self.sess.run([self.yaw_predicted, self.pitch_predicted, self.roll_predicted],
                                feed_dict = train_dict)
                    print('GT: {}\nP :{},{},{}'.format(batch_cont_labels[0], yaw[0], pitch[0], roll[0]))

                if step % 100==0:
                    tf.logging.info('the epoch %d: the loss of the step %d is: total_loss:%f\n loss_yaw:%f\n loss_pitch:%f\n loss_roll:%f'%(epoch, step, loss, yaw_loss, pitch_loss, roll_loss))
                    print('the epoch %d: the loss of the step %d is: total_loss:%f\n loss_yaw:%f\n loss_pitch:%f\n loss_roll:%f'%(epoch, step, loss, yaw_loss, pitch_loss, roll_loss))
                
                if step % 500==0:
                    tf.logging.info('the epoch:%d, save the model for step %d'%(epoch,step))
                    print('the epoch:%d, save the model for step %d'%(epoch,step))
                    saver.save(self.sess, os.path.join(self.args.checkpoint_dir,'model'), global_step=tf.cast(step*epoch, tf.int32))
                    
        tf.logging.info('==================Train Finished================')
        print('==================Train Finished================')


    def infer_testset(self):
        dataset = datasets.Pose_300W_LP(data_dir='./300W_LP',
                                        filename_path=self.args.filename_list,
                                        batch_size=self.args.batch_size,
                                        image_size=self.args.image_size)
        
        for i in range(dataset.length//self.args.batch_size):
            batch_images, batch_labels, batch_cont_labels = dataset.get()
            feed_dict = {self.images: batch_images, 
                            self.is_training: True,
                            self.keep_prob: 0.5}
            yaw, pitch, roll = self.sess.run([self.yaw_predicted,
                                                self.pitch_predicted,
                                                self.roll_predicted],
                                                feed_dict=feed_dict)
            
            tf.logging.info('[] infer test!!!!!\nGT: {}\nP :{},{},{}'.format(batch_cont_labels[0], yaw[0], pitch[0], roll[0]))
            print('[] infer test!!!!!\nGT: {}\nP :{},{},{}'.format(batch_cont_labels[0], yaw[0], pitch[0], roll[0]))
            
            img = datasets.unnomalizing(batch_images[0], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            out = self.draw_axis(img.astype(np.uint8), yaw, pitch, roll)
            cv2.imshow("demo", out)
            cv2.waitKey(0)

        tf.logging.info('[*] test finished')
        print('[*] test finished')


    def load_ckpt(self):
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if (ckpt):
            print('ckpt found...')
            tf.logging.info('restore the trained model')
            saver.restore(self.sess,ckpt)
            return True
        else:
            print('no ckpt')
            tf.logging.info('restore the trained model')
            return False


    def inference(self, input_img):
        
        img = input_img.resize((224,224),Image.BILINEAR)
        # input image
        # format: rgb
        # normalized: [0,1]--> unit gaussian
        precesssed_img = datasets.nomalizing(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # expand dimension
        b_img = np.expand_dims(precesssed_img, axis=0)
    
        feed_dict = {self.images: b_img,
                    self.keep_prob: 1.0,
                    self.is_training: True} # have no idea why this is true

        pre_yaw, pre_pitch, pre_roll = self.sess.run([self.yaw_predicted,
                                                    self.pitch_predicted,
                                                    self.roll_predicted],
                                                    feed_dict=feed_dict)

        yaw, pitch, roll = self.sess.run([self.yaw_predicted,
                                        self.pitch_predicted,
                                        self.roll_predicted],
                                        feed_dict = feed_dict)

        return yaw[0], pitch[0], roll[0]

    @staticmethod
    def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

        return img


class head_pose_estimator():

    def __init__(self, pb_dir='./', pb_filename='./frozen_model.pb'):
        
        self.graph = self.load_graph(os.path.join(pb_dir, pb_filename))

        if False:
            for op in self.graph.get_operations():
                print(op.name)

        # We access the input and output nodes 
        self.keep_prob   = self.graph.get_tensor_by_name('prefix/keep_prob:0')
        self.is_training = self.graph.get_tensor_by_name('prefix/is_training:0')
        self.image_batch = self.graph.get_tensor_by_name('prefix/image_batch:0')
        self.yaw_   = self.graph.get_tensor_by_name('prefix/sub:0')
        self.pitch_ = self.graph.get_tensor_by_name('prefix/sub_1:0')
        self.roll_  = self.graph.get_tensor_by_name('prefix/sub_2:0')
        self.sess = tf.Session(graph=self.graph)


    def __del__(self):
        self.sess.close
        pass


    def inference(self, image):
        # input image
        # format: rgb
        # normalized: [0,1]--> unit gaussian
        image = image.resize((224,224),Image.BILINEAR)
        precesssed_img = datasets.nomalizing(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        b_img = np.expand_dims(precesssed_img, axis=0)

        feed_dict = {
            self.keep_prob: 1.0,
            self.is_training: True,
            self.image_batch: b_img
        }

        yaw, pitch, roll = self.sess.run([self.yaw_, self.pitch_, self.roll_],
                            feed_dict=feed_dict)

        return yaw[0], pitch[0], roll[0]


    def make_pb(self, model_dir):
        '''
        model_dir: path to the ckpt files / Model folder to export
        output_node : The name of the output nodes, comma separated.
        python freeze.py --model_dir ./ckpt-0 --output_node_names sub,sub_1,sub_2
        '''
        output_node_names = 'sub,sub_1,sub_2'        
        self.freeze_graph(model_dir, output_node_names)
        print('pb writing finished !')


    @staticmethod
    def load_graph(frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph


    @staticmethod
    def freeze_graph(model_dir, output_node_names):
        """Extract the sub graph defined by the output nodes and convert 
        all its variables into constant 
        Args:
            model_dir: the root folder containing the checkpoint state file
            output_node_names: a string, containing all the output node's names, 
                                comma separated
        """
        if not tf.gfile.Exists(model_dir):
            raise AssertionError(
                "Export directory doesn't exists. Please specify an export "
                "directory: %s" % model_dir)

        if not output_node_names:
            print("You need to supply the name of a node to --output_node_names.")
            return -1

        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        print('input  checkpoint name : ' + input_checkpoint)

        # We precise the file fullname of our freezed graph
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/frozen_model.pb"

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True

        # We start a session using a temporary fresh Graph
        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

            # We restore the weights
            saver.restore(sess, input_checkpoint)

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                output_node_names.split(",") # The output node names are used to select the usefull nodes
            ) 

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

            # print a list of ops
            for op in output_graph_def.node:
                print(op.name)
            
            print("%d ops in the final graph." % len(output_graph_def.node))

        return output_graph_def

    
if __name__ == '__main__':
    # bool_train = True
    bool_pb_infer = True
    bool_train = False
    # bool_pb_infer = False
    if bool_train:
        args = parse_args()
        net = hopenet(args)
        net.build()
        net.train()

    if bool_pb_infer:
        # load hopenet
        net_hope = head_pose_estimator()

        # load module
        model_dir = './model'
        print('loading tf face detectore module...')
        modelFile = "opencv_face_detector_uint8.pb"
        configFile = "opencv_face_detector.pbtxt"
        modelFile = os.path.join(model_dir, modelFile)
        configFile = os.path.join(model_dir, configFile)
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        print('finished loading tf face detectore module...')
        
        conf_threshold = 0.6
        cap = cv2.VideoCapture(0)
        frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('(w,h): ({},{})'.format(frameWidth,frameHeight))

        if not cap.isOpened():
            print("Unable to connect to camera.")
            exit()
        
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                net.setInput(blob)
                detections = net.forward()
                bboxes = []
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > conf_threshold:

                        x1 = int(detections[0, 0, i, 3] * frameWidth)
                        y1 = int(detections[0, 0, i, 4] * frameHeight)
                        x2 = int(detections[0, 0, i, 5] * frameWidth)
                        y2 = int(detections[0, 0, i, 6] * frameHeight)
                        
                        bboxw = x2 - x1
                        bboxh = y2 - y1

                        x1_pad = int(x1 - bboxw * 0.25)
                        y1_pad = int(y1 - bboxh * 0.25)
                        x2_pad = int(x2 + bboxw * 0.25)
                        y2_pad = int(y2 + bboxh * 0.25)

                        x1_pad = max(x1_pad, 0)
                        y1_pad = max(y1_pad, 0)
                        x2_pad = min(frame.shape[1], x2_pad)
                        y2_pad = min(frame.shape[0], y2_pad)

                        img = rgb[y1_pad:y2_pad, x1_pad:x2_pad]
                        img = Image.fromarray(img)

                        yaw, pitch, roll = net_hope.inference(img)

                        frame = hopenet.draw_axis(frame, yaw, pitch, roll,
                                                tdx = (x1_pad + x2_pad) / 2, tdy= (y1_pad + y2_pad) / 2,
                                                size = bboxh/2)

                        cv2.line(frame, (x1_pad, y1_pad), (x1_pad, y2_pad), (0, 0, 255))
                        cv2.line(frame, (x1_pad, y1_pad), (x2_pad, y1_pad), (0, 0, 255))
                        cv2.line(frame, (x2_pad, y2_pad), (x1_pad, y2_pad), (0, 0, 255))
                        cv2.line(frame, (x2_pad, y2_pad), (x2_pad, y1_pad), (0, 0, 255))

                        cv2.putText(frame, "yaw: " + "{:7.2f}".format(yaw), (20+i*100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), thickness=2)
                        cv2.putText(frame, "pitch: " + "{:7.2f}".format(pitch), (20+i*100, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), thickness=2)
                        cv2.putText(frame, "roll: " + "{:7.2f}".format(roll), (20+i*100, 140), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), thickness=2)

                cv2.imshow("demo", np.array(frame))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break


# tf.logging.set_verbosity(tf.logging.INFO)
# tf.app.run()