
import os
import configparser
from dataset.dataset_container import eval_mode

class validation_config:
    def __init__(self, config_path):

        assert os.path.exists(config_path), 'Configuration file nor found at {}'.format(config_path)
        config = configparser.ConfigParser()
        config.read(config_path)

        self.dataset_config = config.get('Data directory', 'dataset_config')
        self.eval = eval_mode(config.getboolean('Modes', 'eval_vggface2'),
                              config.getboolean('Modes', 'eval_lfw'),
                              config.getboolean('Modes', 'eval_cox'),
                              config.getboolean('Modes', 'eval_chokepoint'))

        self.use_flipped_images = config.getboolean('Modes', 'use_flipped_images')
        self.use_fixed_image_standardization = config.getboolean('Modes', 'use_fixed_image_standardization')
        self.subtract_mean = config.getboolean('Modes', 'subtract_mean')
        self.save_lfw_projections = config.getboolean('Modes', 'save_lfw_projections')
        self.save_cox_projections = config.getboolean('Modes', 'save_cox_projections')

        self.batch_size = config.getint('Hyper parameters', 'batch_size')
        self.image_size = config.getint('Hyper parameters', 'image_size')
        self.nrof_folds = config.getint('Hyper parameters', 'nrof_folds')
        self.distance_metric = config.getint('Hyper parameters', 'distance_metric')
        self.max_emb_per_subject = config.getint('Hyper parameters', 'max_emb_per_subject')
        self.max_subjects = config.getint('Hyper parameters', 'max_subjects')

        self.model = config.get('Data directory', 'model')
        self.emb_dir = config.get('Data directory', 'emb_dir')

class quadruplets_config:
    def __init__(self, config_path):

        assert os.path.exists(config_path), 'Configuration file nor found at {}'.format(config_path)
        config = configparser.ConfigParser()
        config.read(config_path)

        # Modes
        self.random_crop = config.getboolean('Modes', 'random_crop')
        self.random_flip = config.getboolean('Modes', 'random_flip')
        self.eval_lfw = config.getboolean('Modes', 'eval_lfw')
        self.eval_cox = config.getboolean('Modes', 'eval_cox')
        self.save_lfw_projections = config.getboolean('Modes', 'save_lfw_projections')
        self.save_cox_projections = config.getboolean('Modes', 'save_cox_projections')

        # Hyper Parameters
        self.seed = config.getint('Hyper parameters', 'seed')
        self.epoch_size = config.getint('Hyper parameters', 'epoch_size')
        self.max_nrof_epochs = config.getint('Hyper parameters', 'max_nrof_epochs')
        self.batch_size = config.getint('Hyper parameters', 'batch_size')
        self.image_size = config.getint('Hyper parameters', 'image_size')
        self.images_per_person = config.getint('Hyper parameters', 'images_per_person')
        self.people_per_batch = config.getint('Hyper parameters', 'people_per_batch')
        self.nrof_folds = config.getint('Hyper parameters', 'nrof_folds')
        self.keep_probability = config.getfloat('Hyper parameters', 'keep_probability')
        self.embedding_size = config.getint('Hyper parameters', 'embedding_size')
        self.weight_decay = config.getfloat('Hyper parameters', 'weight_decay')
        self.moving_average_decay = config.getfloat('Hyper parameters', 'moving_average_decay')
        self.learning_rate = config.getfloat('Hyper parameters', 'learning_rate')
        self.learning_rate_decay_epochs = config.getint('Hyper parameters', 'learning_rate_decay_epochs')
        self.learning_rate_decay_factor = config.getfloat('Hyper parameters', 'learning_rate_decay_factor')
        self.learning_rate_schedule_file = config.get('Data directory', 'learning_rate_schedule_file')
        self.optimizer = config.get('Hyper parameters', 'optimizer')

        self.alpha = config.getfloat('Hyper parameters', 'alpha')
        self.lamb = config.getfloat('Hyper parameters', 'lamb')
        self.zeta = config.getfloat('Hyper parameters', 'zeta')


        self.gpu_memory_fraction = config.getfloat('Hyper parameters', 'gpu_memory_fraction')

        self.max_emb_per_subject = config.getint('Hyper parameters', 'max_emb_per_subject')
        self.max_subjects = config.getint('Hyper parameters', 'max_subjects')

        # Data Directory
        self.logs_base_dir = config.get('Data directory', 'logs_base_dir')
        self.models_base_dir = config.get('Data directory', 'models_base_dir')
        self.model_def = config.get('Data directory', 'model_def')
        self.pretrained_model = config.get('Data directory', 'pretrained_model')

        self.vggface2_train_dir = config.get('Data directory', 'vggface2_train_dir')
        self.vggface2_val_dir = config.get('Data directory', 'vggface2_val_dir')

        self.lfw_dir = config.get('Data directory', 'lfw_dir')
        self.lfw_pairs = config.get('Data directory', 'lfw_pairs')
        self.lfw_projection = config.get('Data directory', 'lfw_projection')

        self.cox_subject_list = config.get('Data directory', 'cox_subject_list')
        self.cox_still_dir = config.get('Data directory', 'cox_still_dir')
        self.cox_video_dir = config.get('Data directory', 'cox_video_dir')
        self.cox_video_name = config.get('Data directory', 'cox_video_name').split(',')
        self.cox_pairs = config.get('Data directory', 'cox_pairs')
        self.cox_projection = config.get('Data directory', 'cox_projection')

class triplets_afix_config:
    def __init__(self, config_path):

        assert os.path.exists(config_path), 'Configuration file not found at {}'.format(config_path)
        config = configparser.ConfigParser()
        config.read(config_path)

        # Modes
        self.random_crop = config.getboolean('Modes', 'random_crop')
        self.random_flip = config.getboolean('Modes', 'random_flip')
        self.eval_lfw = config.getboolean('Modes', 'eval_lfw')
        self.eval_cox = config.getboolean('Modes', 'eval_cox')
        self.save_lfw_projections = config.getboolean('Modes', 'save_lfw_projections')
        self.save_cox_projections = config.getboolean('Modes', 'save_cox_projections')

        # Hyper Parameters
        self.seed = config.getint('Hyper parameters', 'seed')
        self.epoch_size = config.getint('Hyper parameters', 'epoch_size')
        self.max_nrof_epochs = config.getint('Hyper parameters', 'max_nrof_epochs')
        self.batch_size = config.getint('Hyper parameters', 'batch_size')
        self.image_size = config.getint('Hyper parameters', 'image_size')
        self.images_per_person = config.getint('Hyper parameters', 'images_per_person')
        self.people_per_batch = config.getint('Hyper parameters', 'people_per_batch')
        self.nrof_folds = config.getint('Hyper parameters', 'nrof_folds')
        self.keep_probability = config.getfloat('Hyper parameters', 'keep_probability')
        self.embedding_size = config.getint('Hyper parameters', 'embedding_size')
        self.weight_decay = config.getfloat('Hyper parameters', 'weight_decay')
        self.moving_average_decay = config.getfloat('Hyper parameters', 'moving_average_decay')
        self.learning_rate = config.getfloat('Hyper parameters', 'learning_rate')
        self.learning_rate_decay_epochs = config.getint('Hyper parameters', 'learning_rate_decay_epochs')
        self.learning_rate_decay_factor = config.getfloat('Hyper parameters', 'learning_rate_decay_factor')
        self.learning_rate_schedule_file = config.get('Data directory', 'learning_rate_schedule_file')
        self.optimizer = config.get('Hyper parameters', 'optimizer')

        self.alpha = config.getfloat('Hyper parameters', 'alpha')

        self.gpu_memory_fraction = config.getfloat('Hyper parameters', 'gpu_memory_fraction')

        self.max_emb_per_subject = config.getint('Hyper parameters', 'max_emb_per_subject')
        self.max_subjects = config.getint('Hyper parameters', 'max_subjects')

        # Data Directory
        self.logs_base_dir = config.get('Data directory', 'logs_base_dir')
        self.models_base_dir = config.get('Data directory', 'models_base_dir')
        self.model_def = config.get('Data directory', 'model_def')
        self.pretrained_model = config.get('Data directory', 'pretrained_model')

        # VGGface2
        self.vggface2_train_dir = config.get('Data directory', 'vggface2_train_dir')
        self.vggface2_val_dir = config.get('Data directory', 'vggface2_val_dir')

        # LFW
        self.lfw_dir = config.get('Data directory', 'lfw_dir')
        self.lfw_pairs = config.get('Data directory', 'lfw_pairs')
        self.lfw_projection = config.get('Data directory', 'lfw_projection')

        # COX-S2V
        self.cox_subject_list = config.get('Data directory', 'cox_subject_list')
        self.cox_still_dir = config.get('Data directory', 'cox_still_dir')
        self.cox_video_dir = config.get('Data directory', 'cox_video_dir')
        self.cox_video_name = config.get('Data directory', 'cox_video_name').split(',')
        self.cox_pairs = config.get('Data directory', 'cox_pairs')
        self.cox_projection = config.get('Data directory', 'cox_projection')

        # Chokepoint
        self.chokepoint_still_dir = config.get('Data directory', 'chokepoint_still_dir')
        self.chokepoint_video_dir = config.get('Data directory', 'cochokepoint_video_dirx_pairs')
        self.chokepoint_pairs = config.get('Data directory', 'chokepoint_pairsv')



class dataset:
    def __init__(self, config_path):

        assert os.path.exists(config_path), 'Configuration file not found at {}'.format(config_path)
        config = configparser.ConfigParser()
        config.read(config_path)

        # VGGface2
        self.vggface2_train_dir = config.get('Data directory', 'vggface2_train_dir')
        self.vggface2_val_dir = config.get('Data directory', 'vggface2_val_dir')

        # LFW
        self.lfw_dir = config.get('LFW', 'lfw_dir')
        self.lfw_pairs = config.get('LFW', 'lfw_pairs')
        self.lfw_projection = config.get('LFW', 'lfw_projection')

        # COX-S2V
        self.cox_subject_list = config.get('COX-S2V', 'cox_subject_list')
        self.cox_still_dir = config.get('COX-S2V', 'cox_still_dir')
        self.cox_video_dir = config.get('COX-S2V', 'cox_video_dir')
        self.cox_video_name = config.get('COX-S2V', 'cox_video_name').split(',')
        self.cox_pairs = config.get('COX-S2V', 'cox_pairs')
        self.cox_projection = config.get('COX-S2V', 'cox_projection')

        # Chokepoint
        self.chokepoint_still_dir = config.get('Data directory', 'chokepoint_still_dir')
        self.chokepoint_video_dir = config.get('Data directory', 'cochokepoint_video_dirx_pairs')
        self.chokepoint_pairs = config.get('Data directory', 'chokepoint_pairsv')