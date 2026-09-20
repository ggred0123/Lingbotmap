# Marks this directory as a regular package so it wins over the container-wide
# HuggingFace `datasets` install, which would otherwise shadow it (a namespace
# package loses to a regular package found later on sys.path).
