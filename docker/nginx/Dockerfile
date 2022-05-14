FROM nginx:1.17.6
# Remove default Nginx config
RUN rm /etc/nginx/nginx.conf
# Copy the modified Nginx conf
COPY config/nginx/nginx.conf /etc/nginx
# Copy proxy config
COPY config/nginx/mlflow.conf /etc/nginx/sites-enabled/
