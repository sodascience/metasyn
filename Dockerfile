# use the slim python image 
FROM python:3.11-slim

# Install system dependencies
# git is used by versioneer to define the project version
RUN apt update && apt install -y git

# Install metasyn
COPY . metasyn/
RUN pip install metasyn/

# Remove metasyn folder
RUN rm -r metasyn/

# For excel output use optional XlsxWriter package
RUN pip install XlsxWriter

# Remove system dependencies
RUN apt remove -y git && apt autoremove -y

ENTRYPOINT [ "metasyn" ]
