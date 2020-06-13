
import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task.model_spec import mobilenet_v2_spec, resnet_50_spec

train_data = ImageClassifierDataLoader.from_folder("/home/mouaz/PycharmProjects/Bitirme/dataset-jpeg")
test_data =  ImageClassifierDataLoader.from_folder("/home/mouaz/PycharmProjects/Bitirme/dataset-test-jpeg")

model = image_classifier.create(train_data, batch_size=35, model_spec=resnet_50_spec, epochs=5)
# model = image_classifier.create(train_data, batch_size=35, model_spec=mobilenet_v2_spec, epochs=1)

loss, accuracy = model.evaluate(test_data)

model.export(export_dir='/home/mouaz/PycharmProjects/Bitirme/Python/LiteModel', with_metadata=True)