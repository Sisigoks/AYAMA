import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import Results from './ResultsPage.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Results />
  </StrictMode>,
);
