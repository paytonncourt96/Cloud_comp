# Cloud_comp
This repository contains contents for launching a streamlit webapp via docker container.

To launch on your local machine:
1. Clone the GitHub repository to your local machine using the following command:
   git clone https://github.com/paytonncourt96/Cloud_comp.git

2. Change Directory to project directory:
   cd cloud_comp

3. Build the Docker container using the provided Dockerfile:
      docker build -t cloud_comp_project -f "path\to\your\project\directory\Dockerfile.txt" .

4. Run the Docker Container:
   docker run -p 8501:8501 cloud_comp_project

5. Access the Web App:
   http://localhost:8501

