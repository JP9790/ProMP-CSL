from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'kuka_promp_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='KUKA robot control with ProMP for demonstration learning',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'demo_recorder = kuka_promp_control.demo_recorder:main',
            'interactive_demo_recorder = kuka_promp_control.interactive_demo_recorder:main',
            'train_and_execute = kuka_promp_control.train_and_execute:main',
            'train_promp_only_node = kuka_promp_control.train_promp_only_node:main',
            'standalone_deformation_controller = kuka_promp_control.standalone_deformation_controller:main',
            'airl_controller = kuka_promp_control.AIRL_controller:main',
            'promp_condition_controller = kuka_promp_control.promp_condition_controller:main',
            'promp_inc_controller = kuka_promp_control.promp_inc_controller:main',
            'trajectory_deformer = kuka_promp_control.trajectory_deformer:main',
            'stepwise_em_learner = kuka_promp_control.stepwise_em_learner:main',
            'control_script = kuka_promp_control.control_script:main',
        ],
    },
) 