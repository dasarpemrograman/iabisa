import Navbar from "../components/navbar";
import DashboardLayout from "../components/dashboard-layout";
// sample data for charts

export default function App() {
  return (
    <div className="flex h-screen flex-col bg-white">
      <Navbar />
      <DashboardLayout />
    </div>
  );
}
