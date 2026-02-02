#!/bin/bash

apt-get update
apt-get install -y cmake

##################
# setup egl      #
##################
apt-get install -y \
    libglvnd0 \
    libgl1 \
    libegl1 
    # libgles2 \
    # libglib2.0-dev \
    # libglew2.2 \
    # libnvidia-egl-gbm1 \
    # libnvidia-egl-wayland1 \
    # libnvidia-egl-wayland-dev

mkdir -p /usr/share/glvnd/egl_vendor.d/

# Create a more permissive NVIDIA config that tries multiple library paths
cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json << 'EOF'
{
   "file_format_version" : "1.0.0",
   "ICD" : {
      "library_path" : "libEGL_nvidia.so.0"
   }
}
EOF

# Also ensure Mesa is available as fallback
cat > /usr/share/glvnd/egl_vendor.d/50_mesa.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_mesa.so.0"
    }
}
EOF

ldconfig

##################
# setup libero   #
##################
mkdir -p ~/.libero
touch ~/.libero/config.yaml && \
python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
