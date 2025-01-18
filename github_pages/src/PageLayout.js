import React from 'react';
import Footer from './Footer';

import './PageLayout.css';

const PageLayout = ({ children }) => {
  return (
    <>
      <div className='main-content'>{children}</div>
      <Footer />
    </>
  );
};

export default PageLayout;
