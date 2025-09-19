FROM node:18-alpine AS build
WORKDIR /app
# Copy package manifests (lock file optional)
COPY package*.json ./
# Install dependencies (use lock when present)
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
# Copy source and build
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Nginx config to serve app and proxy /api and /ws to backend service 'web:8000'
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
