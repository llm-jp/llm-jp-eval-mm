import { createRoot } from 'react-dom/client';
import Main from './Main';
import './index.css';

const container = document.querySelector('#root');
const root = createRoot(container);

root.render(<Main />);
