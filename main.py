import argparse
import sys
import model_inputs
import tensorflow as tf
import trainer_functions
import settings
import time
import xception

slim = tf.contrib.slim

def main(_):
  if settings.FLAGS.job_name == "worker" and settings.FLAGS.task_index == 0:
    model_inputs.maybe_download_and_extract()
  ps_hosts = settings.FLAGS.ps_hosts.split(",")
  worker_hosts = settings.FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=settings.FLAGS.job_name,
                           task_index=settings.FLAGS.task_index)

  if settings.FLAGS.job_name == "ps":
    server.join()
  elif settings.FLAGS.job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % settings.FLAGS.task_index,
        cluster=cluster)):
      isXception = trainer_functions.query_yes_no("Would you like to use the Xception model \n(if no, the model will default to that of the TensorFlow turorial)?")
      # Build model
      if isXception:
          images, labels = trainer_functions.distorted_inputs(isXception)
          with slim.arg_scope(xception.xception_arg_scope()):
            logits, end_points = xception.xception(images, num_classes = 10, is_training = True)
      else:
          images, labels = trainer_functions.distorted_inputs(isXception)
          logits = trainer_functions.tutorial_model(images)
      # Calculate loss.
      loss = trainer_functions.loss(logits, labels)
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=settings.FLAGS.max_steps)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(settings.FLAGS.task_index == 0),
                                           checkpoint_dir="./train_logs",
                                           hooks=hooks) as mon_sess:
      prev_time = time.time()
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)
        if mon_sess.run(global_step)%20 == 0:
          duration = time.time() - prev_time
          prev_time = time.time()
          examples_per_sec = settings.FLAGS.log_frequency * settings.FLAGS.batch_size / duration
          print ("examples/sec: %d" % examples_per_sec + ", loss: %f" % mon_sess.run(loss))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_dir', type=str, default='./trainer_functions_train',
                      help='Directory where to write event logs and checkpoint.')

  parser.add_argument('--max_steps', type=int, default=1000000,
                      help='Number of batches to run.')

  parser.add_argument('--log_device_placement', type=bool, default=False,
                      help='Whether to log device placement.')

  parser.add_argument('--log_frequency', type=int, default=10,
                      help='How often to log results to the console.')
  # Basic model parameters.
  parser.add_argument('--batch_size', type=int, default=32,
                      help='Number of images to process in a batch.')

  parser.add_argument('--data_dir', type=str, default='./trainer_functions_data',
                      help='Path to the CIFAR-10 data directory.')

  parser.add_argument('--use_fp16', type=bool, default=False,
                      help='Train the model using fp16.')
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="localhost:2222",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="localhost:2223,localhost:2224",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  settings.FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  """
python trainer.py \
     --job_name=ps --task_index=0

  """