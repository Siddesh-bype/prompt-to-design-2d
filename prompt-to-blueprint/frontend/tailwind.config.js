/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                blueprint: {
                    50: '#E8F4FD',
                    100: '#C5E4F9',
                    200: '#9DD2F5',
                    300: '#74BFF0',
                    400: '#4DABEB',
                    500: '#2196F3',
                    600: '#1976D2',
                    700: '#1565C0',
                    800: '#0D47A1',
                    900: '#0A3781',
                },
                surface: {
                    50: '#FAFBFC',
                    100: '#F4F5F7',
                    200: '#EBEDF0',
                    300: '#DFE1E5',
                    400: '#C1C7D0',
                    500: '#A5AEBB',
                    600: '#7A8699',
                    700: '#5E6C7F',
                    800: '#2C3E50',
                    900: '#1A252F',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
        },
    },
    plugins: [],
};
