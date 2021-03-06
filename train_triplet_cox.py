from src import train_tripletloss_cox
from argparse import Namespace

if __name__ == '__main__':

    args = Namespace(
        alpha=0.2,
        batch_size=45,
        embedding_size=128,
        epoch_size=50,
        gpu_memory_fraction=1.0,
        image_size=160,
        images_per_person=5,
        people_per_batch=9,  # 720
        keep_probability=1.0,
        learning_rate=0.0001,
        learning_rate_decay_epochs=4,
        learning_rate_decay_factor=1.0,
        learning_rate_schedule_file='data/learning_rate_schedule.txt',
        max_nrof_epochs=10,
        model_def='src.models.inception_resnet_v2',
        moving_average_decay=0.9999,
        optimizer='ADAGRAD', #'ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'
        pretrained_model='/export/livia/data/lemoineh/facenet/models/test_1/model-20180917-150357.ckpt-500131',
        random_crop=False,
        random_flip=False,
        seed=666,
        weight_decay=2e-4,

        # Parameters for validation on LFW
        lfw_dir='/export/livia/data/lethanh/lfw/lfw_mtcnnpy_160',
        lfw_nrof_folds=5,
        lfw_pairs='data/pairs.txt',

        # data_dir='/export/livia/Database/COX-S2V/Aligned-COX-S2V-Video/video2',
        # still_dir='/export/livia/data/lemoineh/COX-S2V/COX-S2V-Still-MTCNN160',
        # video_dir='/export/livia/data/lemoineh/COX-S2V/COX-S2V-Video-MTCNN160_MARG44/video2',
        # models_base_dir='/export/livia/data/lemoineh/facenet/COX_finetune/models',
        # logs_base_dir='/export/livia/data/lemoineh/facenet/COX_finetune/logs',
        # cox_pairs = 'data/cox_video2/cox_pairs_40000.txt',

        still_dir='/export/livia/data/lemoineh/ChokePoint/Stills/edit_train_160',
        video_dir='/export/livia/data/lemoineh/ChokePoint/train_rgb',
        models_base_dir='/export/livia/data/lemoineh/facenet/chokepoint_rgb_test2/models',
        logs_base_dir='/export/livia/data/lemoineh/facenet/chokepoint_rgb_test2/logs',
        chokepoint_pairs='chokepoint_pairs.txt'
        #chokepoint_pairs = 'data/chokepoint_rgb_pairs.txt'

    #     train_tripletloss.py.
    #         python
    # src / train_tripletloss.py - -logs_base_dir
    # ~ / logs / facenet / --models_base_dir
    # ~ / models / facenet / --data_dir
    # ~ / datasets / casia / casia_maxpy_mtcnnalign_182_160 - -image_size
    # 160 - -model_def
    # models.inception_resnet_v1 - -lfw_dir
    # ~ / datasets / lfw / lfw_mtcnnalign_160 - -optimizer
    # RMSPROP - -learning_rate
    # 0.01 - -weight_decay
    # 1e-4 - -max_nrof_epochs
    # 500



    )

    train_tripletloss_cox.main(args)