import { Routes, Route, Navigate } from 'react-router-dom';
import RunLoader from './pages/RunLoader';
import InvestigationPage from './pages/InvestigationPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/runs" replace />} />
      <Route path="/runs" element={<RunLoader />} />
      <Route path="/run/:runId" element={<InvestigationPage />} />
    </Routes>
  );
}

export default App;
