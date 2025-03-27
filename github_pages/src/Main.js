import BibTex from './BibTex';
// import Example from "./Example";
import Introduction from './Introduction';
import Leaderboard from './Leaderboard';
import PaperMetaData from './PaperMetaData';

import PageLayout from './PageLayout';
import './Main.css';

const Main = () => {
  return (
    <PageLayout>
      <PaperMetaData />
      <Introduction />
      {/* <Method /> */}
      {/* <Example /> */}
      <Leaderboard />
      {/* <Result /> */}
      <BibTex />
    </PageLayout>
  );
};

export default Main;
