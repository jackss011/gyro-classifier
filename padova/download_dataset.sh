#!/bin/bash

# dataset used by the students
gdown https://docs.google.com/uc?id=1yPNWWeubU9SjRKqW7H4-q0BaNsw57jWc -O ../dataset/dataset1.zip

# the other datasets provided by the authors of the paper
#gdown https://docs.google.com/uc?id=1ET_YY8CYQTKQb1mvSd4EAQuv4OymEBAZ
#gdown https://docs.google.com/uc?id=137LunCBjT-dThC9p03eNbySb89D6XbFk
#gdown https://docs.google.com/uc?id=1EA9gIHsGSAnkyjbYin8n9-1YTCzEMbXT
#gdown https://docs.google.com/uc?id=1dmfl3-24Vy2QAa9IQ4ccjLyFVLIeMbTS
#gdown https://docs.google.com/uc?id=1rLAg5ngzuQKa0eL1DAVFtNY1xNcteLKB
#gdown https://docs.google.com/uc?id=1H0hEY4kSvGFZTejA7nYFwd3EZBIDa1ql
#gdown https://docs.google.com/uc?id=1QVPy-fq9yJKQ60DdN1n4CL0CQAGJjnpU

unzip ../dataset/dataset1.zip -d ../dataset
mv ../dataset/Dataset\ \#1 ../dataset/dataset1
rm ../dataset/dataset1.zip

