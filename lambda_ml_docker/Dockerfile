FROM lambci/lambda:build-python3.6

RUN curl https://nodejs.org/dist/v8.2.1/node-v8.2.1-linux-x64.tar.xz | tar -xJ -C /opt && pushd . && \
  ln -fs /opt/node-v8.2.1-linux-x64/bin/* /usr/bin/.
ENV PATH=/opt/node-v8.2.1-linux-x64/bin/:${PATH}

RUN chmod --recursive 777 /tmp && groupadd -g 1000 lambdadev \
        && adduser -u 1000 -g lambdadev -s /bin/bash lambdadev

RUN pip install pipenv && pip install -U awscli

RUN mkdir /lambda_ml_docker
COPY . /lambda_ml_docker/
RUN cd /lambda_ml_docker && pipenv install --dev --system && pipenv install --system && npm install -g serverless

RUN chown -R lambdadev:lambdadev /lambda_ml_docker
USER lambdadev

CMD ["/bin/bash"]
