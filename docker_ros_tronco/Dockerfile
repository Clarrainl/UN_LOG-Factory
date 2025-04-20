# Imagen base con ROS Noetic y Python3
FROM ros:noetic-ros-base

# Configuración del entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Actualizaciones e instalación de dependencias
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-numpy \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libosmesa6-dev \
    libglfw3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear directorios
WORKDIR /root/catkin_ws/src

# Inicializar workspace ROS
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_init_workspace"

# Copiar el código ROS (esto se ajusta en build time)
COPY ../ /root/catkin_ws/src/log_defect_detector

# Compilar el workspace
WORKDIR /root/catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Instalar Python dependencias
WORKDIR /root
RUN pip3 install --upgrade pip
RUN pip3 install \
    opencv-python \
    pandas \
    ultralytics \
    vedo \
    trimesh \
    PyOpenGL \
    PyOpenGL_accelerate

# Set ROS environment on start
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

# Arranque por defecto: bash interactivo
CMD ["/bin/bash"]
