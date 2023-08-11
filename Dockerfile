# use the slim python image 
FROM python:3.11-slim

# Install system dependencies
# git is used by versioneer to define the project version
RUN apt update && apt install -y git

# Install metasynth
COPY . metasynth/
RUN pip install metasynth/

# Remove metasynth folder
RUN rm -r metasynth/

# For excel output use optional XlsxWriter package
RUN pip install XlsxWriter

# Remove system dependencies
RUN apt remove -y git && apt autoremove -y

ENTRYPOINT [ "metasynth" ]
