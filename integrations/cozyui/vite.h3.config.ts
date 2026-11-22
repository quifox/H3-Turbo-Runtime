import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

const proxy = {
    '/cui': {
        target: process.env.COMFYUI_URL ?? 'http://127.0.0.1:8188',
        ws: true,
        rewrite: (path: string) => path.replace(/^\/cui/, ''),
    },
};

export default defineConfig({
    plugins: [react()],
    base: './',
    server: { host: '127.0.0.1', proxy },
    preview: { host: '127.0.0.1', proxy },
});
