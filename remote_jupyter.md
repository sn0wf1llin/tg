ssh -i ~/Documents/work/kaf/ekeys/e_cloud_dev.pem ec2-user@52.14.6.38

ssh -N -f -L localhost:8899:localhost:65000 -i ~/Documents/work/kaf/ekeys/e_cloud_dev.pem ec2-user@52.14.6.38

