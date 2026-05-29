# ==================================================================================
#
#       Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==================================================================================
"""kfadapter_kfconnect.py.

This module is for interfacing and interworking with KubeFlow SDK

"""
import kfp

from kfadapter.kfadapter_util import random_suffix
from kfadapter.kfadapter_conf import KfConfiguration

class KfConnect:
    """
    This is a class for interfacing and interworking with KubeFlow SDK.

    Attributes: None
    """
    def __init__(self):
        """
        The constructor for KfConnect class.

        Parameters:None
         """
        self.kfp_client = None
        kfc_config = KfConfiguration.get_instance()
        self.logger = kfc_config.logger
        self.logger.debug("Initialized KfConnect")

    def set_kf_client(self, kfp_client):
        self.kfp_client = kfp_client

    def get_kf_client(self, host=None):
        """
        Function for getting kubeflow connection handle for use

        Args:
            host: hostname or host ip where kubeflow is running

        Returns: None

        """
        self.kfp_client = kfp.Client(host)

    def get_kf_list_experiments(self, nspace):
        """
        Function for getting list of experiments based on namespace

        Args:
            nspace: namespace from which experiments can be listed

        Returns: experiment list

        """
        exp = self.kfp_client.list_experiments(page_size=10)
        return exp

    def get_kf_experiment_details(self, ex_name, nspace):
        """
        Function for getting experiment description based on experiment name

        Args:
            ex_name: experiment name
            nspace: namespace from which experiments can be listed

        Returns:
                experiment details

        """
        self.logger.debug("Get Experiment details " + ex_name)
        try:
            exp = self.kfp_client.get_experiment(experiment_name=ex_name)
        except ValueError as err:
            self.logger.error(err)
            return None

        return exp

    def get_kf_run(self, run_id):
        """
        Function for getting status of run based on run_id

        Args:
            run_id: run_id for a previous run

        Returns: run status

        """
        run = self.kfp_client.get_run(run_id)
        return run

    def get_kf_list_runs(self, nspace):
        """
        Function for getting list of runs based on namespace

        Args:
            nspace: namespace from which runs can be listed

        Returns: runs list

        """
        runs = self.kfp_client.list_runs(page_size=20)
        return runs

    def get_kf_list_pipelines(self):
        """
        Function for getting list of pipelines in kubeflow

        Args: None

        Returns: list of pipeline and its description

        """
        pipeline = self.kfp_client.list_pipelines(page_size=20)
        return pipeline

    def get_kf_pipeline_id(self, pipeline_name):
        """
        Function for getting pipeline id for a pipeline name in kubeflow

        Args:
            pipeline_name: name of pipeline

        Returns:pipeline id in a string

        """
        pipe_id = self.kfp_client.get_pipeline_id(pipeline_name)
        return pipe_id

    def get_kf_pipeline_version_id(self, pipeline_id, pipeline_version_name):
        """
        Function for getting pipeline's version id for a pipeline's version name
        for given pipeline id in kubeflow

        Args:
            pipeline_id: id of pipeline
            pipeline_version_name: name of pipeline's version

        Returns:pipeline's version id in a string

        """
        version_id = None

        obj_list = self.kfp_client.list_pipeline_versions(pipeline_id,
                                                          page_size=1000000000).pipeline_versions
        for pipeline_version_obj in obj_list:
            if pipeline_version_obj.display_name == pipeline_version_name:
                version_id = pipeline_version_obj.pipeline_version_id
        return version_id

    def upload_kf_pipeline(self, pipeline_name, file, desc):
        """
        Function for uploading pipeline in kubeflow

        Args:
            pipeline_name: name of pipeline
            file: zip file containing pipeline code

        Returns:pipeline id in a string

        """
        pipe_info = self.kfp_client.upload_pipeline(file, pipeline_name, desc)
        return pipe_info

    def upload_pipeline_with_versions(self, pipeline_name, file, desc):
        """
        Function for uploading pipeline with version functionality

        Args:
            pipeline_name: name of pipeline
            file: zip file containing pipeline code
            desc: description of pipeline

        Returns:object containing pipleine id and other information

        """
        versions_list = self.get_pl_versions_by_pl_name(pipeline_name)
        length = len(versions_list)
        pipe_info = None
        if length == 0:
            pipe_info = self.kfp_client.upload_pipeline(file, pipeline_name=pipeline_name,
                                                        description=desc)
        else:
            pipe_info = self.kfp_client.upload_pipeline_version(file,
                                                                pipeline_version_name=str(length+1),
                                                                pipeline_name=pipeline_name)
        return pipe_info

    def get_pl_versions_by_pl_name(self, pipeline_name):
        """
        Function for getting versions list for given pipeline name

        Args:
            pipeline_name: name of pipeline

        Returns:list containing versions name

        """
        pipeline_id = self.get_kf_pipeline_id(pipeline_name)
        if pipeline_id == None:
            return []
        res_obj = self.kfp_client.list_pipeline_versions(pipeline_id,
                                                         page_size=1000000000)
        if res_obj.total_size is None:
            return []
        obj_list = res_obj.pipeline_versions
        versions_list = []
        for obj in obj_list:
            versions_list.append(obj.display_name)
        return versions_list

    def get_kf_pipeline_desc(self, pipeline_id):
        """
        Function for getting pipeline description for a pipeline id in kubeflow

        Args:
            pipeline_id: id of pipeline

        Returns:pipeline description

        """
        pipeline = self.kfp_client.get_pipeline(pipeline_id)
        return pipeline

    def delete_kf_pipeline(self, pipeline_id):
        """
        Function for deleting pipeline for a pipeline id in kubeflow

        Args:
            pipeline_id: id of pipeline

        Returns:pipeline description

        """
        pipeline = self.kfp_client.delete_pipeline(pipeline_id)
        return pipeline

    def run_kf_pipeline(self, exp_id, pipeline_id, arguments, version_id):
        """
        Function for running pipeline with arguments under an experiment in kubeflow

        Args:
            exp_id: experiment under which pipeline run will occur
            pipeline_id: id of pipeline
            arguments: arguments to the pipeline
            version id: version of the pipeline
        Returns:run description

        """
        self.logger.debug("run_kf_pipeline Entered")
        job_id=arguments["trainingjob_id"]
        featuregroup_name = arguments["featuregroup_name"]
        featurepath = featuregroup_name+"_"+job_id
        req_dict={
            "featurepath":featurepath,
            "epochs": arguments["epochs"],
            "modelname": str(arguments["modelName"]),
            "modelversion": str(arguments["modelVersion"])
        }
        self.logger.debug("run_kf_pipeline Arguments: "+str(req_dict))
        try:
            run = self.kfp_client.run_pipeline(exp_id, job_name="testjob"+\
                                           random_suffix(),
                                           pipeline_package_path=None, params=req_dict,
                                           pipeline_id=pipeline_id,
                                           version_id=version_id)
        except Exception as err:
            self.logger.error(str(err))

        self.logger.debug("run_kf_pipeline Exited")
        return run
    
    
    def terminate_kf_pipeline(self, run_id):
        print("Terminating Run: run_id: ", run_id)
        try:
            out = self.kfp_client.terminate_run(run_id)
            print("Terminate Run O/p :: ", out)
        except Exception as err:
            self.logger.error("Terminate Run Error :: ", str(err))
            raise err