import { defineConfig } from 'vite';

export default () => {
  return defineConfig({
    // Serve files from the frontend directory
    root: 'frontend',
    envDir: '../',
    base: './',
    server: {
      port: 5173,
    },
    build: {
      // Output the built assets to the top-level "dist" folder
      outDir: '../dist',
      emptyOutDir: true,
      rollupOptions: {
        input: {
          main: 'frontend/index.html',
          dashboard: 'frontend/dashboard.html',
        },
      },
    },
  });
};
