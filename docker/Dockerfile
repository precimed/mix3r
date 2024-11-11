FROM mambaorg/micromamba:2.0-cuda12.4.1-ubuntu22.04

WORKDIR /tmp/install
COPY ../environment.yml .
RUN micromamba env update --name base --file environment.yml && \
    micromamba clean --all --yes

WORKDIR /tmp
RUN rm -rf install

WORKDIR /scripts
COPY ../mix3r_int_weights.py .
COPY ../extract_p.py .
COPY ../make_euler.r .
COPY ../make_template.py .

WORKDIR /

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
