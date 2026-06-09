import { Routes, Route, Navigate } from 'react-router-dom';
import RunLoader from './pages/RunLoader';
import InvestigationPage from './pages/InvestigationPage';
import RunComparison from './pages/RunComparison';
import BenchmarkDashboard from './pages/BenchmarkDashboard';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/runs" replace />} />
      <Route path="/runs" element={<RunLoader />} />
      <Route path="/run/:runId" element={<InvestigationPage />} />
      <Route path="/compare/:runIdA/:runIdB" element={<RunComparison />} />
      <Route path="/benchmark" element={<BenchmarkDashboard />} />
    </Routes>
  );
}

export default App;
