import tensorflow as tf
import keras
from keras.models import load_model


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def retrieve_ctc_model():
    model = load_model("CTCPRIMUS1K0agnostic.h5")
    input_layer = model.get_layer('the_input')
    pred_layer = model.get_layer('softmax')

    
retrieve_ctc_model()



