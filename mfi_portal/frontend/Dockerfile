FROM node:20-alpine AS build

WORKDIR /app

COPY mfi_portal/frontend/package.json mfi_portal/frontend/package-lock.json ./
RUN npm ci

COPY mfi_portal/frontend/ .
RUN npm run build

FROM nginx:alpine

COPY mfi_portal/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 5174

CMD ["nginx", "-g", "daemon off;"]
