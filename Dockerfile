# use the slim python image 
FROM python:3.11-slim

# Install system dependencies
# git is used by versioneer to define the project version
RUN apt update && apt install -y git

# Install metasyn
COPY . metasyn/
RUN pip install ./metasyn[extra]

# Remove metasyn folder
RUN rm -r metasyn/

# Remove system dependencies
RUN apt remove -y git && apt autoremove -y

ENTRYPOINT [ "metasyn" ]
