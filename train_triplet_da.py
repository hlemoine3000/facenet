from src import train_tripletloss_da
from argparse import Namespace

if __name__ == '__main__':

    args = Namespace(
        alpha=0.2,
        batch_size=72,
        embedding_size=512,
        epoch_size=1000,
        gpu_memory_fraction=1.0,
        image_size=160,
        images_per_person=5,
        keep_probability=1.0,
        lamb=0.1, # Adverserial loss weight
        zeta=1.0, # Triplet loss weight
        learning_rate=0.0005,
        learning_rate_decay_epochs=4,
        learning_rate_decay_factor=0.98,
        learning_rate_schedule_file='data/learning_rate_schedule.txt',
        max_nrof_epochs=1000,
        model_def='src.models.inception_resnet_v2',
        moving_average_decay=0.9999,
        optimizer='ADAGRAD', #'ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'
        people_per_batch=720, # 720
        pretrained_model='/export/livia/data/lemoineh/facenet/models/test_3/model-20181001-122238.ckpt-500514',
        random_crop=False,
        random_flip=False,
        seed=666,
        weight_decay=2e-4,

        vggface2_train_dir='/export/livia/data/lemoineh/vggface2/train_182',
        vggface2_val_dir='/export/livia/data/lemoineh/vggface2/test_182',

        # Parameters for validation on LFW
        lfw_dir='/export/livia/data/lethanh/lfw/lfw_mtcnnpy_160',
        lfw_nrof_folds=10,
        lfw_pairs='data/pairs.txt',
        lfw_projection='data/lfw_projection.txt',

        # data_dir='/export/livia/Database/COX-S2V/Aligned-COX-S2V-Video/video2',
        cox_still_dir='/export/livia/data/lemoineh/COX-S2V/COX-S2V-Still-MTCNN160',
        cox_video_dir='/export/livia/data/lemoineh/COX-S2V/COX-S2V-Video-MTCNN160_MARG44/video2',
        cox_pairs='data/cox/video2/pairs.txt',
        cox_projection='data/cox/cox_projection.txt',

        models_base_dir='/export/livia/data/lemoineh/facenet/test/models',
        logs_base_dir='/export/livia/data/lemoineh/facenet/test/logs'


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

    train_tripletloss_da.main(args)