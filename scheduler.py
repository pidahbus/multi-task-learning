import tensorflow as tf

class LinearLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, df, batch_size, num_epochs, initial_lr):
        super().__init__()
    
        self.num_steps = tf.cast((df.shape[0]*1.1 // batch_size) * num_epochs, tf.float32)
        self.initial_lr = tf.cast(initial_lr, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        
    
        return self.initial_lr * (self.num_steps - step)/self.num_steps