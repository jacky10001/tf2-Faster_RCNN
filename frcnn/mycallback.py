# -*- coding: utf-8 -*-
"""

@author: Jacky Gao
@date: Fri Dec 18 05:28:15 2020
"""

import tensorflow as tf
from pushbullet import Pushbullet


class send_train_peogress_to_pushbullet(tf.keras.callbacks.Callback):
    def __init__(self, set_name, send_freq=1):
        super().__init__()
        self.set_name = set_name
        self.send_freq = send_freq
        
        api_key = "o.bW72HxQhL0Ylr53B5JpWzj4crLVXfdyT"
        self.pb = Pushbullet(api_key)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        title = "Training %s - epoch %d"%(self.set_name,int(epoch)+1)
        
        msg = ""
        for k, v in logs.items():
            msg += "{}: {}\n".format(k,v)
        
        if epoch%self.send_freq == 0:
            push = self.pb.push_note(title, msg)
        self.logs = logs
            
    def on_train_end(self, logs=None):
        msg = ""
        for k, v in self.logs.items():
            msg += "{}: {}\n".format(k,v)
            
        push = self.pb.push_note("%s Model"%self.set_name,
                                 "Training Complete.\n"+msg)
            
