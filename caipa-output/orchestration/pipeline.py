import kfp
import kfp.components as comp
import kfp.dsl as dsl
from kfp.gcp import use_gcp_secret
from kfp.components import ComponentStore
from os import path
import json

cs = ComponentStore(local_search_paths=['.', './caipa-output'],
                    url_search_prefixes=['https://raw.githubusercontent.com/kubeflow/pipelines/3f4b80127f35e40760eeb1813ce1d3f641502222/components/gcp/'])

#cs.local_search_paths.append('')

preprocess_op = cs.load_component('user-input/preprocess')
hpt_op = cs.load_component('hptune')
param_comp = cs.load_component('get_tuned_params')
train_op = cs.load_component('ml_engine/train')
deploy_op = cs.load_component('ml_engine/deploy')

@dsl.pipeline(
    name='KFP-Pipelines Example',
    description='Kubeflow pipeline generated from ai-pipeline asset'
)
def pipeline_sample(
   project_id='gcp-ml-demo-233523',
   region = 'us-central1',
   python_module = 'trainer.task',
   package_uri = 'gs://poc-bucket-0120/trainer.tar.gz',
   dataset_bucket = 'poc-bucket-0120',
   staging_bucket = 'gs://poc-bucket-0120',
   job_dir_hptune = 'gs://poc-bucket-0120/hptune',
   job_dir_train = 'gs://poc-bucket-0120/train',
   runtime_version_train = '1.10',
   runtime_version_deploy = '1.10',
   hptune_config='gs://poc-bucket-0120/hpconfig.yaml',
   model_id='Loand_Delinq',
   version_id='v1.0',
   common_args_hpt=json.dumps([
                   '--output_dir', 'gs://poc-bucket-0120/hptune' ,
                   '--input_bucket', 'gs://poc-bucket-0120' ,
                   '--eval_steps', '10' ,
                   '--train_examples', '200' ,
   ]),
   common_args_train=json.dumps([
                   '--output_dir', 'gs://poc-bucket-0120/train' ,
                   '--input_bucket', 'gs://poc-bucket-0120' ,
                   '--eval_steps', '10' ,
                   '--train_examples', '2000' ,
   ]),
   replace_existing_version=True
):

    #Preprocess Task
    preprocess_task = preprocess_op(
        project_id=project_id,
        dataset_bucket=dataset_bucket,
     )

    # HP tune Task
    hpt_task = hpt_op (
         region=region,
         python_module=python_module,
         package_uri=package_uri,
         staging_bucket=staging_bucket,
         job_dir=job_dir_hptune,
         config=hptune_config,
         runtime_version=runtime_version_train,
         args=common_args_hpt ,
    )
    hpt_task.after(preprocess_task)

    # Get the best hyperparameters
    param_task = param_comp (
        project_id=project_id,
        hptune_job_id=hpt_task.outputs['job_id'].to_struct(),
        common_args=common_args_train,
    )

    # Train Task
    train_task = train_op (
        project_id=project_id,
        python_module=python_module,
        package_uris=json.dumps([package_uri.to_struct()]),
        region=region,
        args=str(param_task.outputs['tuned_parameters_out']) ,
        job_dir=job_dir_train,
        python_version='',
        runtime_version=runtime_version_train,
        master_image_uri='',
        worker_image_uri='',
        training_input='',
        job_id_prefix='',
        wait_interval='30'
    )

         #model_uri=train_task.outputs['job_dir'],
         #model_uri='gs://poc-bucket-0120/train/out/export/exporter',
    deploy_model = deploy_op(
         model_uri=train_task.outputs['job_dir'].to_struct()+'/export/exporter',
         project_id=project_id,
         model_id=model_id,
         version_id=version_id,
         runtime_version=runtime_version_deploy,
         replace_existing_version=replace_existing_version
    )
    #deploy_model.after(train_task)



    kfp.dsl.get_pipeline_conf().add_op_transformer(use_gcp_secret('user-gcp-sa'))

client = kfp.Client(host='https://1f1cbba3c89218d3-dot-us-central2.pipelines.googleusercontent.com')

client.create_run_from_pipeline_func(pipeline_sample, arguments={})

