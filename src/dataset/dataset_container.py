import os
from dataset import lfw, cox, chokepoint, vggface2
import configparser

class eval_mode:
    def __init__(self,
                 get_vggface2=False,
                 get_lfw=False,
                 get_cox_s2v=False,
                 get_chokepoint=False):

        self.vggface2 = get_vggface2
        self.lfw = get_lfw
        self.cox_s2v = get_cox_s2v
        self.chokepoint = get_chokepoint

class dataset_container:
    def __init__(self,
                 config_path,
                 mode: eval_mode):

        assert os.path.exists(config_path), 'Configuration file not found at {}'.format(config_path)
        config = configparser.ConfigParser()
        config.read(config_path)

        # VGGface2
        self.vggface2_dataset = None
        if mode.vggface2:

            self.vggface2_dataset = vggface2.vggface2_data(config.get('VGGface2', 'train_dir'),
                                                           config.get('VGGface2', 'test_dir'),
                                                           config.get('VGGface2', 'pairs'))


        # LFW
        self.lfw_dataset = None
        self.lfw_projection = None
        if mode.lfw:
            self.lfw_dataset = lfw.lfw_data(config.get('LFW', 'test_dir'),
                                            config.get('LFW', 'lfw_pairs'))
            self.lfw_projection = config.get('LFW', 'projection')

        # COX-S2V
        self.cox_dataset = None
        self.cox_subject_list = None
        self.cox_video_name = None
        self.cox_projection = None
        if mode.cox_s2v:

            self.cox_dataset = cox.cox_data(config.get('COX-S2V', 'still_dir'),
                                            config.get('COX-S2V', 'video_dir'),
                                            config.get('COX-S2V', 'pairs'))

            self.cox_subject_list = config.get('COX-S2V', 'cox_subject_list')
            self.cox_video_name = config.get('COX-S2V', 'cox_video_name').split(',')
            self.cox_projection = config.get('COX-S2V', 'cox_projection')

        # Chokepoint
        self.chokepoint_dataset = None
        if mode.chokepoint:

            self.chokepoint_dataset = chokepoint.chokepoint_data(config.get('Chokepoint', 'still_dir'),
                                                                 config.get('Chokepoint', 'video_dir'),
                                                                 config.get('Chokepoint', 'pairs'))