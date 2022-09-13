
from autogbm.utils.log_utils import logger
 
from .auto_tabular.model import Model as TabularModel

class Model:

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        self.done_training = False
        self.metadata = metadata
        self.domain = "tabular"
        DomainModel = TabularModel
        self.domain_model = DomainModel(self.metadata)
        self.has_exception = False
        self.y_pred_last = None

    def save_model(self):
        """
        :return (model_obj, config_obj)
        """
        logger.info("start to save model")
        return self.domain_model.save_model(modal_type=self.domain)

    def load_model(self, model_obj, config_obj):
        logger.info("start to load model")
        self.domain_model.load_model(model_obj, config_obj)

    def fit(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        # Convert training dataset to necessary format and
        # store as self.domain_dataset_train

        try:
            self.domain_model.fit(dataset, remaining_time_budget)
            self.done_training = self.domain_model.done_training

        except Exception as exp:
            logger.exception("exception in fit: {}".format(exp))
            self.has_exception = True
            self.done_training = True

    def predict(self, dataset, remaining_time_budget=None, test=False):
        """Test method of domain-specific model."""
        # Convert test dataset to necessary format and
        # store as self.domain_dataset_test
        # Make predictions

        if self.done_training is True or self.has_exception is True:
            return self.y_pred_last

        try:
            Y_pred = self.domain_model.predict(dataset, remaining_time_budget=remaining_time_budget, test=test)

            self.y_pred_last = Y_pred
            self.done_training = self.domain_model.done_training

        except MemoryError as mem_error:
            logger.exception("exception in predict: {}".format(mem_error))
            self.has_exception = True
            self.done_training = True
        except Exception as exp:
            logger.exception("exception in predict: {}".format(exp))
            self.has_exception = True
            self.done_training = True

        return self.y_pred_last
