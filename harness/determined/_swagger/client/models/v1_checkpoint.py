# coding: utf-8

"""
    Determined API (Beta)

    Determined helps deep learning teams train models more quickly, easily share GPU resources, and effectively collaborate. Determined allows deep learning engineers to focus on building and training models at scale, without needing to worry about DevOps or writing custom code for common tasks like fault tolerance or experiment tracking.  You can think of Determined as a platform that bridges the gap between tools like TensorFlow and PyTorch --- which work great for a single researcher with a single GPU --- to the challenges that arise when doing deep learning at scale, as teams, clusters, and data sets all increase in size.  # noqa: E501

    OpenAPI spec version: 0.1
    Contact: community@determined.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six


class V1Checkpoint(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'uuid': 'str',
        'experiment_config': 'object',
        'experiment_id': 'int',
        'trial_id': 'int',
        'hparams': 'object',
        'batch_number': 'int',
        'start_time': 'datetime',
        'end_time': 'datetime',
        'resources': 'dict(str, str)',
        'metadata': 'object',
        'framework': 'str',
        'format': 'str',
        'determined_version': 'str',
        'metrics': 'V1Metrics',
        'validation_state': 'Determinedcheckpointv1State',
        'state': 'Determinedcheckpointv1State',
        'searcher_metric': 'float'
    }

    attribute_map = {
        'uuid': 'uuid',
        'experiment_config': 'experimentConfig',
        'experiment_id': 'experimentId',
        'trial_id': 'trialId',
        'hparams': 'hparams',
        'batch_number': 'batchNumber',
        'start_time': 'startTime',
        'end_time': 'endTime',
        'resources': 'resources',
        'metadata': 'metadata',
        'framework': 'framework',
        'format': 'format',
        'determined_version': 'determinedVersion',
        'metrics': 'metrics',
        'validation_state': 'validationState',
        'state': 'state',
        'searcher_metric': 'searcherMetric'
    }

    def __init__(self, uuid=None, experiment_config=None, experiment_id=None, trial_id=None, hparams=None, batch_number=None, start_time=None, end_time=None, resources=None, metadata=None, framework=None, format=None, determined_version=None, metrics=None, validation_state=None, state=None, searcher_metric=None):  # noqa: E501
        """V1Checkpoint - a model defined in Swagger"""  # noqa: E501

        self._uuid = None
        self._experiment_config = None
        self._experiment_id = None
        self._trial_id = None
        self._hparams = None
        self._batch_number = None
        self._start_time = None
        self._end_time = None
        self._resources = None
        self._metadata = None
        self._framework = None
        self._format = None
        self._determined_version = None
        self._metrics = None
        self._validation_state = None
        self._state = None
        self._searcher_metric = None
        self.discriminator = None

        if uuid is not None:
            self.uuid = uuid
        if experiment_config is not None:
            self.experiment_config = experiment_config
        self.experiment_id = experiment_id
        self.trial_id = trial_id
        if hparams is not None:
            self.hparams = hparams
        self.batch_number = batch_number
        self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
        if resources is not None:
            self.resources = resources
        if metadata is not None:
            self.metadata = metadata
        if framework is not None:
            self.framework = framework
        if format is not None:
            self.format = format
        if determined_version is not None:
            self.determined_version = determined_version
        if metrics is not None:
            self.metrics = metrics
        if validation_state is not None:
            self.validation_state = validation_state
        self.state = state
        if searcher_metric is not None:
            self.searcher_metric = searcher_metric

    @property
    def uuid(self):
        """Gets the uuid of this V1Checkpoint.  # noqa: E501

        UUID of the checkpoint.  # noqa: E501

        :return: The uuid of this V1Checkpoint.  # noqa: E501
        :rtype: str
        """
        return self._uuid

    @uuid.setter
    def uuid(self, uuid):
        """Sets the uuid of this V1Checkpoint.

        UUID of the checkpoint.  # noqa: E501

        :param uuid: The uuid of this V1Checkpoint.  # noqa: E501
        :type: str
        """

        self._uuid = uuid

    @property
    def experiment_config(self):
        """Gets the experiment_config of this V1Checkpoint.  # noqa: E501

        The configuration of the experiment that created this checkpoint.  # noqa: E501

        :return: The experiment_config of this V1Checkpoint.  # noqa: E501
        :rtype: object
        """
        return self._experiment_config

    @experiment_config.setter
    def experiment_config(self, experiment_config):
        """Sets the experiment_config of this V1Checkpoint.

        The configuration of the experiment that created this checkpoint.  # noqa: E501

        :param experiment_config: The experiment_config of this V1Checkpoint.  # noqa: E501
        :type: object
        """

        self._experiment_config = experiment_config

    @property
    def experiment_id(self):
        """Gets the experiment_id of this V1Checkpoint.  # noqa: E501

        The ID of the experiment that created this checkpoint.  # noqa: E501

        :return: The experiment_id of this V1Checkpoint.  # noqa: E501
        :rtype: int
        """
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, experiment_id):
        """Sets the experiment_id of this V1Checkpoint.

        The ID of the experiment that created this checkpoint.  # noqa: E501

        :param experiment_id: The experiment_id of this V1Checkpoint.  # noqa: E501
        :type: int
        """
        if experiment_id is None:
            raise ValueError("Invalid value for `experiment_id`, must not be `None`")  # noqa: E501

        self._experiment_id = experiment_id

    @property
    def trial_id(self):
        """Gets the trial_id of this V1Checkpoint.  # noqa: E501

        The ID of the trial that created this checkpoint.  # noqa: E501

        :return: The trial_id of this V1Checkpoint.  # noqa: E501
        :rtype: int
        """
        return self._trial_id

    @trial_id.setter
    def trial_id(self, trial_id):
        """Sets the trial_id of this V1Checkpoint.

        The ID of the trial that created this checkpoint.  # noqa: E501

        :param trial_id: The trial_id of this V1Checkpoint.  # noqa: E501
        :type: int
        """
        if trial_id is None:
            raise ValueError("Invalid value for `trial_id`, must not be `None`")  # noqa: E501

        self._trial_id = trial_id

    @property
    def hparams(self):
        """Gets the hparams of this V1Checkpoint.  # noqa: E501

        Hyperparameter values for the trial that created this checkpoint.  # noqa: E501

        :return: The hparams of this V1Checkpoint.  # noqa: E501
        :rtype: object
        """
        return self._hparams

    @hparams.setter
    def hparams(self, hparams):
        """Sets the hparams of this V1Checkpoint.

        Hyperparameter values for the trial that created this checkpoint.  # noqa: E501

        :param hparams: The hparams of this V1Checkpoint.  # noqa: E501
        :type: object
        """

        self._hparams = hparams

    @property
    def batch_number(self):
        """Gets the batch_number of this V1Checkpoint.  # noqa: E501

        Batch number of this checkpoint.  # noqa: E501

        :return: The batch_number of this V1Checkpoint.  # noqa: E501
        :rtype: int
        """
        return self._batch_number

    @batch_number.setter
    def batch_number(self, batch_number):
        """Sets the batch_number of this V1Checkpoint.

        Batch number of this checkpoint.  # noqa: E501

        :param batch_number: The batch_number of this V1Checkpoint.  # noqa: E501
        :type: int
        """
        if batch_number is None:
            raise ValueError("Invalid value for `batch_number`, must not be `None`")  # noqa: E501

        self._batch_number = batch_number

    @property
    def start_time(self):
        """Gets the start_time of this V1Checkpoint.  # noqa: E501

        Timestamp when the checkpoint began being saved to persistent storage.  # noqa: E501

        :return: The start_time of this V1Checkpoint.  # noqa: E501
        :rtype: datetime
        """
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        """Sets the start_time of this V1Checkpoint.

        Timestamp when the checkpoint began being saved to persistent storage.  # noqa: E501

        :param start_time: The start_time of this V1Checkpoint.  # noqa: E501
        :type: datetime
        """
        if start_time is None:
            raise ValueError("Invalid value for `start_time`, must not be `None`")  # noqa: E501

        self._start_time = start_time

    @property
    def end_time(self):
        """Gets the end_time of this V1Checkpoint.  # noqa: E501

        Timestamp when the checkpoint completed being saved to persistent storage.  # noqa: E501

        :return: The end_time of this V1Checkpoint.  # noqa: E501
        :rtype: datetime
        """
        return self._end_time

    @end_time.setter
    def end_time(self, end_time):
        """Sets the end_time of this V1Checkpoint.

        Timestamp when the checkpoint completed being saved to persistent storage.  # noqa: E501

        :param end_time: The end_time of this V1Checkpoint.  # noqa: E501
        :type: datetime
        """

        self._end_time = end_time

    @property
    def resources(self):
        """Gets the resources of this V1Checkpoint.  # noqa: E501

        Dictionary of file paths to file sizes in bytes of all files in the checkpoint.  # noqa: E501

        :return: The resources of this V1Checkpoint.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._resources

    @resources.setter
    def resources(self, resources):
        """Sets the resources of this V1Checkpoint.

        Dictionary of file paths to file sizes in bytes of all files in the checkpoint.  # noqa: E501

        :param resources: The resources of this V1Checkpoint.  # noqa: E501
        :type: dict(str, str)
        """

        self._resources = resources

    @property
    def metadata(self):
        """Gets the metadata of this V1Checkpoint.  # noqa: E501

        User defined metadata associated with the checkpoint.  # noqa: E501

        :return: The metadata of this V1Checkpoint.  # noqa: E501
        :rtype: object
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """Sets the metadata of this V1Checkpoint.

        User defined metadata associated with the checkpoint.  # noqa: E501

        :param metadata: The metadata of this V1Checkpoint.  # noqa: E501
        :type: object
        """

        self._metadata = metadata

    @property
    def framework(self):
        """Gets the framework of this V1Checkpoint.  # noqa: E501

        The framework of the trial i.e., tensorflow, torch.  # noqa: E501

        :return: The framework of this V1Checkpoint.  # noqa: E501
        :rtype: str
        """
        return self._framework

    @framework.setter
    def framework(self, framework):
        """Sets the framework of this V1Checkpoint.

        The framework of the trial i.e., tensorflow, torch.  # noqa: E501

        :param framework: The framework of this V1Checkpoint.  # noqa: E501
        :type: str
        """

        self._framework = framework

    @property
    def format(self):
        """Gets the format of this V1Checkpoint.  # noqa: E501

        The format of the checkpoint i.e., h5, saved_model, pickle.  # noqa: E501

        :return: The format of this V1Checkpoint.  # noqa: E501
        :rtype: str
        """
        return self._format

    @format.setter
    def format(self, format):
        """Sets the format of this V1Checkpoint.

        The format of the checkpoint i.e., h5, saved_model, pickle.  # noqa: E501

        :param format: The format of this V1Checkpoint.  # noqa: E501
        :type: str
        """

        self._format = format

    @property
    def determined_version(self):
        """Gets the determined_version of this V1Checkpoint.  # noqa: E501

        The version of Determined the checkpoint was taken with.  # noqa: E501

        :return: The determined_version of this V1Checkpoint.  # noqa: E501
        :rtype: str
        """
        return self._determined_version

    @determined_version.setter
    def determined_version(self, determined_version):
        """Sets the determined_version of this V1Checkpoint.

        The version of Determined the checkpoint was taken with.  # noqa: E501

        :param determined_version: The determined_version of this V1Checkpoint.  # noqa: E501
        :type: str
        """

        self._determined_version = determined_version

    @property
    def metrics(self):
        """Gets the metrics of this V1Checkpoint.  # noqa: E501

        Dictionary of validation metric names to their values.  # noqa: E501

        :return: The metrics of this V1Checkpoint.  # noqa: E501
        :rtype: V1Metrics
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """Sets the metrics of this V1Checkpoint.

        Dictionary of validation metric names to their values.  # noqa: E501

        :param metrics: The metrics of this V1Checkpoint.  # noqa: E501
        :type: V1Metrics
        """

        self._metrics = metrics

    @property
    def validation_state(self):
        """Gets the validation_state of this V1Checkpoint.  # noqa: E501

        The state of the validation associated with this checkpoint.  # noqa: E501

        :return: The validation_state of this V1Checkpoint.  # noqa: E501
        :rtype: Determinedcheckpointv1State
        """
        return self._validation_state

    @validation_state.setter
    def validation_state(self, validation_state):
        """Sets the validation_state of this V1Checkpoint.

        The state of the validation associated with this checkpoint.  # noqa: E501

        :param validation_state: The validation_state of this V1Checkpoint.  # noqa: E501
        :type: Determinedcheckpointv1State
        """

        self._validation_state = validation_state

    @property
    def state(self):
        """Gets the state of this V1Checkpoint.  # noqa: E501

        The state of the checkpoint.  # noqa: E501

        :return: The state of this V1Checkpoint.  # noqa: E501
        :rtype: Determinedcheckpointv1State
        """
        return self._state

    @state.setter
    def state(self, state):
        """Sets the state of this V1Checkpoint.

        The state of the checkpoint.  # noqa: E501

        :param state: The state of this V1Checkpoint.  # noqa: E501
        :type: Determinedcheckpointv1State
        """
        if state is None:
            raise ValueError("Invalid value for `state`, must not be `None`")  # noqa: E501

        self._state = state

    @property
    def searcher_metric(self):
        """Gets the searcher_metric of this V1Checkpoint.  # noqa: E501

        The value of the metric specified by `searcher.metric` for this metric.  # noqa: E501

        :return: The searcher_metric of this V1Checkpoint.  # noqa: E501
        :rtype: float
        """
        return self._searcher_metric

    @searcher_metric.setter
    def searcher_metric(self, searcher_metric):
        """Sets the searcher_metric of this V1Checkpoint.

        The value of the metric specified by `searcher.metric` for this metric.  # noqa: E501

        :param searcher_metric: The searcher_metric of this V1Checkpoint.  # noqa: E501
        :type: float
        """

        self._searcher_metric = searcher_metric

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(V1Checkpoint, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1Checkpoint):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other