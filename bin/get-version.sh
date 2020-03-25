#!/usr/bin/env bash

set -e

version=$(grep "version = " setup.cfg)
version=${version/version = }
version=${version/\'/}
version=${version/\'/}
version=${version/\"/}
version=${version/\"/}

echo $version
