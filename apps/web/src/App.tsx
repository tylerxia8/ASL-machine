import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import { useAuth } from "./lib/auth";
import LoginPage from "./pages/LoginPage";
import LobbyPage from "./pages/LobbyPage";
import PracticePage from "./pages/PracticePage";
import ProgressPage from "./pages/ProgressPage";
import CapturePage from "./pages/CapturePage";
import DryRunPage from "./pages/DryRunPage";
import PhrasePage from "./pages/PhrasePage";
import ReviewCapturesPage from "./pages/ReviewCapturesPage";

function Protected({ children }: { children: React.ReactNode }) {
  const { loading } = useAuth();
  if (loading) return <div className="container">Loading…</div>;
  return <>{children}</>;
}

export default function App() {
  const { user, signOut, devUserId } = useAuth();
  const loggedIn = !!user || !!devUserId;

  return (
    <>
      <nav className="nav">
        <strong>ASL Practice Pilot</strong>
        {loggedIn && (
          <>
            <NavLink to="/lobby">Practice</NavLink>
            <NavLink to="/progress">Progress</NavLink>
            <NavLink to="/capture">Capture</NavLink>
            <NavLink to="/review-captures">Review</NavLink>
            <NavLink to="/phrases">Phrases</NavLink>
            <NavLink to="/dry-run">Dry run</NavLink>
            <span style={{ marginLeft: "auto", color: "var(--muted)", fontSize: "0.85rem" }}>
              {user?.email ?? `Dev: ${devUserId.slice(0, 12)}`}
            </span>
            <button className="btn btn-secondary" onClick={() => signOut()}>
              Sign out
            </button>
          </>
        )}
      </nav>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/lobby"
          element={
            <Protected>
              <LobbyPage />
            </Protected>
          }
        />
        <Route
          path="/practice"
          element={
            <Protected>
              <PracticePage />
            </Protected>
          }
        />
        <Route
          path="/progress"
          element={
            <Protected>
              <ProgressPage />
            </Protected>
          }
        />
        <Route
          path="/capture"
          element={
            <Protected>
              <CapturePage />
            </Protected>
          }
        />
        <Route
          path="/review-captures"
          element={
            <Protected>
              <ReviewCapturesPage />
            </Protected>
          }
        />
        <Route
          path="/phrases"
          element={
            <Protected>
              <PhrasePage />
            </Protected>
          }
        />
        <Route
          path="/dry-run"
          element={
            <Protected>
              <DryRunPage />
            </Protected>
          }
        />
        <Route path="/" element={<Navigate to={loggedIn ? "/lobby" : "/login"} replace />} />
      </Routes>
    </>
  );
}
