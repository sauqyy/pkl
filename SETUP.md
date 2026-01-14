# Setup Guide - React Frontend Migration

## Overview
The application now has a **React frontend** that communicates with the existing Flask backend via API endpoints.

## Architecture
- **Backend (Flask)**: Runs on `http://localhost:5000` - serves API endpoints
- **Frontend (React + Vite)**: Runs on `http://localhost:5173` - serves the UI

---

## Setup Instructions

### 1. Pull Latest Changes
```bash
git pull origin main
```

### 2. Backend Setup (Flask)
The backend setup remains the same:

```bash
# Install Python dependencies (if not already installed)
pip install -r requirements.txt

# Run the Flask server
python app.py
```

The backend will be available at `http://localhost:5000`

### 3. Frontend Setup (React)
This is **NEW** - you need to set up the React frontend:

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies (first time only)
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

---

## Running the Application

### Option 1: Two Terminal Windows (Recommended for Development)

**Terminal 1 - Backend:**
```bash
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Option 2: Background Processes
You can run the backend in the background and frontend in the foreground:

**Windows:**
```bash
# Start backend in background
start python app.py

# Start frontend
cd frontend
npm run dev
```

---

## Accessing the Application

1. **Open your browser** and go to: `http://localhost:5173`
2. The React app will automatically proxy API requests to the Flask backend at `http://localhost:5000`

### Available Pages
- `/` - Executive Dashboard (System Overview)
- `/response-time` - Response Time Analysis
- `/forecasting` - AI Performance Forecast
- `/load-analysis` - Load Analysis
- `/error-analysis` - Error Analysis
- `/slow-calls-analysis` - Slow Calls Analysis

---

## Prerequisites

### Required Software
- **Python 3.x** (for Flask backend)
- **Node.js 18+** (for React frontend)
- **npm** (comes with Node.js)

### Installing Node.js
If you don't have Node.js installed:
- Download from: https://nodejs.org/
- Choose the LTS (Long Term Support) version
- Verify installation: `node --version` and `npm --version`

---

## Troubleshooting

### Port Already in Use
If port 5173 is already in use:
```bash
# The Vite dev server will automatically try the next available port
# Check the terminal output for the actual port being used
```

If port 5000 is already in use:
```bash
# Stop the other process using port 5000, or
# Edit app.py to use a different port
```

### CORS Issues
The frontend is configured to proxy requests to the backend. If you see CORS errors:
- Make sure Flask-CORS is installed: `pip install flask-cors`
- Verify the backend is running on port 5000

### Dependencies Issues
If `npm install` fails:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and try again
rm -rf node_modules
npm install
```

---

## Development Workflow

1. **Start both servers** (backend and frontend)
2. **Make changes** to the code
3. **Hot reload** - Both servers support hot reload:
   - React: Changes appear instantly in the browser
   - Flask: Restart may be needed for some changes
4. **Test** your changes at `http://localhost:5173`

---

## Build for Production

To create a production build of the frontend:

```bash
cd frontend
npm run build
```

This creates an optimized build in `frontend/dist/` directory.

---

## Notes

- The **Flask server must be running** for the frontend to fetch data
- The React dev server **proxies API requests** to Flask automatically
- All existing Flask routes (`/api/*`) remain unchanged
- HTML templates in `templates/` are now replaced by React components

---

## Questions?

If you encounter any issues:
1. Check that both servers are running
2. Verify Node.js and Python are properly installed
3. Clear browser cache and restart dev servers
4. Check terminal output for error messages
